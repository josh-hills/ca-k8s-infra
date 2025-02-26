.PHONY: entrypoint install train deploy start_port_forward stop_port_forward

entrypoint: deploy train

make refresh:
	@echo "Refreshing resources..."
	@helm uninstall raycluster
	@helm upgrade --install raycluster ./infra/helm/raycluster 

deploy:
	@echo "Deploying resources..."
	@helm upgrade --install raycluster ./infra/helm/raycluster
	@echo "Waiting for jobs to complete..."
	@kubectl get jobs -o name | xargs -I {} bash -c 'kubectl wait --for=condition=complete {} --timeout=300s || kubectl wait --for=condition=failed {} --timeout=300s'


start_port_forward:
	@echo "Starting port forwarding..."
	@kubectl port-forward service/raycluster-head-svc 8265:8265 10001:10001 & echo $$! >> .port_forward.pid

stop_port_forward:
	@echo "Stopping port forwarding..."
	@if [ -f .port_forward.pid ]; then \
		kill `cat .port_forward.pid`; rm .port_forward.pid; \
	else \
		echo "No port forwarding process found"; \
	fi

install:
	uv sync

train: install start_port_forward
	@trap 'make stop_port_forward' EXIT 2 15; \
	echo "Training model..." && \
	RAY_AIR_NEW_OUTPUT=0 uv run python training/train.py
