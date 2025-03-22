import os
import tempfile
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow
import ray
import torch
from datasets import load_from_disk
from pyarrow.fs import S3FileSystem
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

runtime_env = {
    "env_vars": {
        "ENVIRONMENT": os.environ["ENVIRONMENT"],
    }
}

# Define the scaling configuration
scaling_config = ScalingConfig(
    num_workers=2,
    use_gpu=False,
    resources_per_worker={"CPU": 1, "memory": 4 * 1024 * 1024 * 1024},
)

# Training configuration
train_config = {
    "model_location": "pythia-70m",  # This is the path in MinIO under the models bucket
    "batch_size": 2,
    "learning_rate": 5e-5,
    "num_epochs": 1,
    "max_iterations": 10,
    # More realistic sample size
    "train_samples": 4,
    "test_samples": 2,
}


# For copying from S3 to local:
def copy_dir_from_s3(fs, source_path, target_path):
    file_info = fs.get_file_info(pyarrow.fs.FileSelector(source_path))
    for file in file_info:
        if file.type == pyarrow.fs.FileType.Directory:
            subdir = os.path.basename(file.path)
            os.makedirs(os.path.join(target_path, subdir), exist_ok=True)
            copy_dir_from_s3(fs, file.path, os.path.join(target_path, subdir))
        else:
            os.makedirs(target_path, exist_ok=True)
            with (
                fs.open_input_file(file.path) as s3_file,
                open(
                    os.path.join(target_path, os.path.basename(file.path)), "wb"
                ) as local_file,
            ):
                local_file.write(s3_file.read())


# For copying from local to S3:
def copy_dir_to_s3(fs, source_path, target_path):
    for root, _, files in os.walk(source_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_path)
            s3_path = os.path.join(target_path, relative_path)
            with (
                open(local_path, "rb") as local_file,
                fs.open_output_stream(s3_path) as s3_file,
            ):
                s3_file.write(local_file.read())


# Simple function to calculate perplexity from loss
def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()


def train_func(config):
    # Track training metrics
    train_losses = []
    perplexities = []
    step_times = []
    test_losses = []

    start_time = time.time()
    rank = ray.train.get_context().get_world_rank()

    # Initialize the S3 filesystem
    fs = S3FileSystem(
        access_key="minioadmin",
        secret_key="minioadmin",  # noqa
        endpoint_override=f"minio.minio-internal-{os.environ['ENVIRONMENT']}.svc.cluster.local:9000",
        scheme="http",
    )

    # Copy data to disk
    print("Copying data to disk")
    with TemporaryDirectory() as tmp_dir:
        imdb_path = Path(tmp_dir)
        copy_dir_from_s3(fs, "datasets/imdb/", imdb_path)
        print("Data copied to disk")

        # Load datasets
        train_dataset = load_from_disk(str(imdb_path / "train"))
        test_dataset = load_from_disk(str(imdb_path / "test"))

    # Take subset of data
    train_samples = config.get("train_samples", 2)
    test_samples = config.get("test_samples", 2)

    print(f"Train dataset size: {len(train_dataset)}, using {train_samples} samples")
    print(f"Test dataset size: {len(test_dataset)}, using {test_samples} samples")

    train_dataset = train_dataset.take(train_samples)
    test_dataset = test_dataset.take(test_samples)

    # Prepare local directory for the model files
    s3_path = f"models/{config['model_location']}/"

    with tempfile.TemporaryDirectory() as local_model_dir:
        # Copy the model files from S3 to local directory
        print(f"Copying model files from {s3_path} to {local_model_dir}")
        copy_dir_from_s3(fs, s3_path, local_model_dir)
        print("Model files copied to local directory")

        # Initialize tokenizer and model from local directory
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            local_model_dir, use_safetensors=True
        )
        print(
            f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters"
        )
        model.gradient_checkpointing_enable()
        model.use_cache = False

        # Prepare the model for distributed training
        model = prepare_model(model)

        def preprocess_function(examples):
            prompts = [
                f"Review: {text}\nSentiment: {label}\n"
                for text, label in zip(examples["text"], examples["label"], strict=True)
            ]
            encoded = tokenizer(
                prompts, truncation=True, max_length=512, padding="max_length"
            )
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded

        # Preprocess datasets
        print("Preprocessing datasets...")
        tokenized_train = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,  # pyright: ignore
        )
        tokenized_test = test_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,  # pyright: ignore
        )

        # Convert to PyTorch datasets
        train_dataset = tokenized_train.with_format("torch")
        test_dataset = tokenized_test.with_format("torch")

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config["batch_size"], shuffle=False
        )

        # Prepare data loaders for distributed training
        train_loader = prepare_data_loader(train_loader)
        test_loader = prepare_data_loader(test_loader)

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

        # Training loop
        model.train()
        epoch_loss = 0.0
        global_step = 0

        for epoch in range(config["num_epochs"]):
            epoch_loss = 0.0
            print(f"Starting epoch {epoch + 1}/{config['num_epochs']}")

            for batch in train_loader:
                step_start_time = time.time()
                optimizer.zero_grad()

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                # Calculate metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss
                step_time = time.time() - step_start_time
                perplexity = calculate_perplexity(batch_loss)

                # Track metrics
                train_losses.append(batch_loss)
                perplexities.append(perplexity)
                step_times.append(step_time)

                # Log step metrics
                print(
                    f"Epoch {epoch + 1}, Step {global_step + 1}: Loss = {batch_loss:.4f}, "
                    f"Perplexity = {perplexity:.2f}, Step time = {step_time:.2f}s"
                )

                # Report metrics to Ray
                ray.train.report(
                    {
                        "training_loss": batch_loss,
                        "epoch": epoch,
                        "perplexity": perplexity,
                        "step_time": step_time,
                    }
                )

                global_step += 1

            # Evaluate on test set
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(**batch)
                    test_loss += outputs.loss.item()

            avg_test_loss = test_loss / len(test_loader)
            test_perplexity = calculate_perplexity(avg_test_loss)
            test_losses.append(avg_test_loss)

            # Log eval metrics
            print(
                f"Epoch {epoch + 1} eval: Test Loss = {avg_test_loss:.4f}, "
                f"Test Perplexity = {test_perplexity:.2f}"
            )

            model.train()

            # Report test loss
            ray.train.report(
                {
                    "test_loss": avg_test_loss,
                    "test_perplexity": test_perplexity,
                    "epoch": epoch,
                }
            )

        # Save the model on the main process
        if rank == 0:
            # Save the model
            if isinstance(model, DistributedDataParallel):
                model.module.save_pretrained(local_model_dir)
            else:
                model.save_pretrained(local_model_dir)
            tokenizer.save_pretrained(local_model_dir)

            # Copy the saved model files back to S3
            checkpoint_path = f"models/checkpoints/sentiment_model/{ray.train.get_context().get_trial_id()}"
            copy_dir_to_s3(fs, local_model_dir, checkpoint_path)

            # Print summary of metrics
            print("\nTraining Summary:")
            print(f"Total training time: {time.time() - start_time:.2f}s")
            print(f"Average training loss: {sum(train_losses) / len(train_losses):.4f}")
            print(f"Average perplexity: {sum(perplexities) / len(perplexities):.2f}")
            print(f"Average step time: {sum(step_times) / len(step_times):.2f}s")
            if test_losses:
                print(f"Final test loss: {test_losses[-1]:.4f}")
                print(
                    f"Final test perplexity: {calculate_perplexity(test_losses[-1]):.2f}"
                )

    # Return metrics for Ray
    return {
        "final_loss": epoch_loss / len(train_loader) if len(train_loader) > 0 else 0,
        "final_test_loss": test_losses[-1] if test_losses else 0,
        "final_perplexity": calculate_perplexity(test_losses[-1]) if test_losses else 0,
        "training_time": time.time() - start_time,
    }


if __name__ == "__main__":
    # Run `make ray_port_forward` ahead of this to make the ray cluster accessible
    ray.init(address="ray://localhost:10001", runtime_env=runtime_env)

    with TemporaryDirectory() as results_dir:
        # Create the trainer
        trainer = TorchTrainer(
            train_func,
            train_loop_config=train_config,
            scaling_config=scaling_config,
            run_config=RunConfig(results_dir),
        )

    # Start training
    try:
        results = trainer.fit()
        print("\nTraining Results:")
        for key, value in results.metrics.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Training failed with error: {e}")
    finally:
        ray.shutdown()
