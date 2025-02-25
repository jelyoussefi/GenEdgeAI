# Settings
SHELL := /bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PORT ?= 80

# Default Models and Precisions
MODELS ?= TinyLlama/TinyLlama-1.1B-Chat-v1.0
#deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
PRECISIONS ?= FP16 INT8 INT4


# Docker Configuration
DOCKER_IMAGE_NAME := gen_age_ai
export DOCKER_BUILDKIT := 1

DOCKER_RUN_PARAMS := \
    -it --rm \
    -a stdout -a stderr \
    --privileged \
    -v /dev:/dev \
    -v $(CURRENT_DIR):/workspace \
    -w /workspace \
    -p $(PORT):$(PORT) \
    -e http_proxy=$(HTTP_PROXY) \
    -e https_proxy=$(HTTPS_PROXY) \
    -e no_proxy=$(NO_PROXY) \
    $(DOCKER_IMAGE_NAME)

DOCKER_BUILD_PARAMS := \
	--rm \
     --build-arg MODELS="$(MODELS)" \
    --build-arg PRECISIONS="$(PRECISIONS)" \
    --build-arg http_proxy=$(HTTP_PROXY) \
    --build-arg https_proxy=$(HTTPS_PROXY) \
    --build-arg no_proxy=$(NO_PROXY) \
    -t $(DOCKER_IMAGE_NAME) . 

# Targets
.PHONY: default build run bash

default: run

build:
	@echo "📦 Building Docker image $(DOCKER_IMAGE_NAME)..."
	@docker build ${DOCKER_BUILD_PARAMS}

run: build
	@echo "🚀 Running Vision Edge AI demo..."
	@docker run $(DOCKER_RUN_PARAMS) bash -c "python3 ./app.py --port $(PORT)"

bash: build
	@echo "🐚 Starting bash in container..."
	@docker run $(DOCKER_RUN_PARAMS) bash
