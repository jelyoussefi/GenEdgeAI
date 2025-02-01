
#----------------------------------------------------------------------------------------------------------------------
# Flags
#----------------------------------------------------------------------------------------------------------------------
SHELL:=/bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Proxy settings (inherited from the host environment if set)
HTTP_PROXY := $(HTTP_PROXY)
HTTPS_PROXY := $(HTTPS_PROXY)
NO_PROXY := $(NO_PROXY)

PORT ?= 80

MODELS ?= \
	"TinyLlama/TinyLlama-1.1B-Chat-v1.0"
#----------------------------------------------------------------------------------------------------------------------
# Docker Settings
#----------------------------------------------------------------------------------------------------------------------
DOCKER_IMAGE_NAME=gen_age_ai
export DOCKER_BUILDKIT=1

# Docker run parameters with proxy settings
DOCKER_RUN_PARAMS= \
	-it --rm -a stdout -a stderr   \
	--privileged -v /dev:/dev \
	-v ${CURRENT_DIR}:/workspace \
	-w /workspace \
	-p ${PORT}:${PORT} \
	-e http_proxy=${HTTP_PROXY} \
	-e https_proxy=${HTTPS_PROXY} \
	-e no_proxy=${NO_PROXY} \
	${DOCKER_IMAGE_NAME}

#----------------------------------------------------------------------------------------------------------------------
# Targets
#----------------------------------------------------------------------------------------------------------------------
default: run
.PHONY: models

build:
	@$(call msg, Building Docker image ${DOCKER_IMAGE_NAME} ...)
	@docker build --rm . -t ${DOCKER_IMAGE_NAME} \
		--build-arg MODELS=${MODELS} \
		--build-arg http_proxy=${HTTP_PROXY} \
		--build-arg https_proxy=${HTTPS_PROXY} \
		--build-arg no_proxy=${NO_PROXY} 
	

run: build
	@$(call msg, Running the Vision Edge AI demo ...)
	@docker run ${DOCKER_RUN_PARAMS} bash -c ' \
		python3 ./app.py --port ${PORT} '

bash: build
	@docker run ${DOCKER_RUN_PARAMS} bash 

#----------------------------------------------------------------------------------------------------------------------
# Helper Functions
#----------------------------------------------------------------------------------------------------------------------
define msg
	tput setaf 2 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo  "" && \
	echo "         "$1 && \
	for i in $(shell seq 1 120 ); do echo -n "-"; done; echo "" && \
	tput sgr0
endef

