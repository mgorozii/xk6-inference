# xk6-inference/examples/Makefile

.PHONY: help up-triton down-triton up-kserve-mm down-kserve-mm tests

help:
	@echo "Usage:"
	@echo "  make up-triton      - Start Triton server"
	@echo "  make down-triton    - Stop Triton server"
	@echo "  make up-kserve-mm      - Start KServe (ModelMesh)"
	@echo "  make down-kserve-mm    - Stop KServe"
	@echo "  make tests          - Run k6 tests"

up-triton:
	cd triton && docker-compose up -d

down-triton:
	cd triton && docker-compose down

up-kserve-mm:
	cd kserve-mm && make up

down-kserve-mm:
	cd kserve-mm && make down

tests:
	@echo "Running tests against Triton..."
	HTTP_URL=http://localhost:8000 GRPC_URL=localhost:8001 MODEL_NAME=simple ./../k6 run test.js
	@echo "Running tests against KServe..."
	HTTP_URL=http://localhost:8008 GRPC_URL=localhost:8003 MODEL_NAME=triton-model ./../k6 run test.js
