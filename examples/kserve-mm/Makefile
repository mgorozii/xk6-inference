.PHONY: up down status inference health metadata server-info logs

NAMESPACE=modelmesh-serving
REPO_DIR=modelmesh-serving-repo
PATH := $(shell pwd):$(PATH)

up:
	@echo "Checking for kustomize..."
	@if ! command -v kustomize >/dev/null 2>&1; then \
		echo "Kustomize not found, downloading to current directory..."; \
		curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash; \
	fi
	@echo "Creating namespace $(NAMESPACE) if not exists..."
	kubectl create namespace $(NAMESPACE) --dry-run=client -o yaml | kubectl apply -f -
	@echo "Checking if $(REPO_DIR) exists..."
	@if [ ! -d "$(REPO_DIR)" ]; then \
		git clone -b main --depth 1 https://github.com/kserve/modelmesh-serving.git $(REPO_DIR); \
	fi
	@echo "Installing ModelMesh..."
	cd $(REPO_DIR) && export PATH=$(shell pwd):$$PATH && ./scripts/install.sh --namespace-scope-mode --namespace $(NAMESPACE) --quickstart --enable-self-signed-ca
	@echo "Waiting for ModelMesh and dependencies to be ready..."
	kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=modelmesh-controller -n $(NAMESPACE) --timeout=300s
	kubectl wait --for=condition=ready pod -l app=minio -n $(NAMESPACE) --timeout=300s
	kubectl wait --for=condition=ready pod -l app=etcd -n $(NAMESPACE) --timeout=300s
	@echo "Deploying sample model..."
	kubectl apply -f manifests/triton-model.yaml -n $(NAMESPACE)
	@echo "Waiting for model to be ready..."
	@echo "Note: This may take a few minutes for the runtime pod to pull the image (~6GB)."
	kubectl wait --for=condition=Ready inferenceservice/triton-model -n $(NAMESPACE) --timeout=600s
	chmod +x scripts/inference.sh scripts/api_call.sh

down:
	@echo "Removing ModelMesh and models..."
	kubectl delete namespace $(NAMESPACE) --ignore-not-found
	rm -rf $(REPO_DIR) kustomize

status:
	kubectl get pods -n $(NAMESPACE)
	kubectl get inferenceservice -n $(NAMESPACE)

# API & Metadata
health:
	@./scripts/api_call.sh model-ready

metadata:
	@./scripts/api_call.sh model-metadata

server-info:
	@./scripts/api_call.sh server-info

inference:
	@echo "Performing inference..."
	@./scripts/inference.sh

logs:
	kubectl logs -l name=modelmesh-serving-triton-2.x -n $(NAMESPACE) -c triton --tail=100
