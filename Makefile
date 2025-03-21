# Settings
SHELL := /bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
CACHE_DIR := $(CURRENT_DIR)/.cache
MODELS_DIR := /workspace/models
PORT ?= 80

# Default Models and Precisions
MODELS ?=  \
		meta-llama/Meta-Llama-3-8B \
		deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
		TinyLlama/TinyLlama-1.1B-Chat-v1.0 
	
PRECISIONS ?= INT4 INT8 

# Docker Configuration
DOCKER_IMAGE_NAME := gen_age_ai
export DOCKER_BUILDKIT := 1

DOCKER_RUN_PARAMS := \
	-it --rm \
	--network=host \
	-a stdout -a stderr \
	--privileged \
	-v /dev:/dev \
	-v $(CURRENT_DIR):/workspace \
	-v $(CACHE_DIR):/root/.cache \
	-w /workspace \
	-p $(PORT):$(PORT) \
	-e http_proxy=$(HTTP_PROXY) \
	-e https_proxy=$(HTTPS_PROXY) \
	-e no_proxy=$(NO_PROXY) \
	$(DOCKER_IMAGE_NAME)

DOCKER_BUILD_PARAMS := \
	--rm \
	--network=host \
	--build-arg MODELS="$(MODELS)" \
	--build-arg PRECISIONS="$(PRECISIONS)" \
	--build-arg HF_TOKEN="$(HF_TOKEN)" \
	--build-arg http_proxy=$(HTTP_PROXY) \
	--build-arg https_proxy=$(HTTPS_PROXY) \
	--build-arg no_proxy=$(NO_PROXY) \
	-t $(DOCKER_IMAGE_NAME) . 

# Targets
.PHONY: default build run bash models

default: run

build:
	@echo "📦 Building Docker image $(DOCKER_IMAGE_NAME)..."
	@mkdir -p -m 777 $(CACHE_DIR) 
	@docker build ${DOCKER_BUILD_PARAMS}

run: 	models
	@echo "🚀 Running Gen Edge AI demo..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c "python3 ./app.py --port $(PORT) --models_dir $(MODELS_DIR)"

models: build
	@docker run $(DOCKER_RUN_PARAMS) ./utils/generate_models.sh $(HF_TOKEN) "$(MODELS)" "$(PRECISIONS)" $(MODELS_DIR)

bash: models
	@echo "🐚 Starting bash in container..."
	@docker run $(DOCKER_RUN_PARAMS) bash
