.PHONY: entrypoint install train deploy start_port_forward stop_port_forward

entrypoint: deploy train

make refresh:
	@echo "Refreshing resources..."
	@kubectl delete -f infra/manifests/
	@kubectl apply -f infra/manifests/

deploy:
	@echo "Deploying resources..."
	@kubectl apply -f infra/manifests/

start_port_forward:
	@echo "Starting port forwarding..."
	@kubectl port-forward --namespace training service/raycluster-head-svc 8265:8265 10001:10001 & echo $$! > .port_forward.pid
	@kubectl port-forward --namespace kube-system service/hubble-ui 8080:80 & echo $$! >> .port_forward.pid

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
