.PHONY: entrypoint install train deploy start_port_forward stop_port_forward

entrypoint: deploy train

make refresh:
	@echo "Refreshing resources..."
	@kubectl --context kind-k8s-code-sabotage delete -f infra/manifests/
	@kubectl --context kind-k8s-code-sabotage apply -f infra/manifests/

deploy:
	@echo "Deploying resources..."
	@kubectl --context kind-k8s-code-sabotage apply -f infra/manifests/
	@echo "Waiting for jobs to complete..."
	@kubectl --context kind-k8s-code-sabotage get jobs -o name | xargs -I {} bash -c 'kubectl --context kind-k8s-code-sabotage wait --for=condition=complete {} --timeout=300s || kubectl --context kind-k8s-code-sabotage wait --for=condition=failed {} --timeout=300s'


start_port_forward:
	@echo "Starting port forwarding..."
	@kubectl --context kind-k8s-code-sabotage port-forward --namespace training service/raycluster-head-svc 8265:8265 10001:10001 & echo $$! >> .port_forward.pid
	@kubectl --context kind-k8s-code-sabotage port-forward --namespace kube-system service/hubble-ui 8080:80 & echo $$! >> .port_forward.pid

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
