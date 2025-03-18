FROM ubuntu:24.10

ARG DEBIAN_FRONTEND=noninteractive

USER root

# Install system dependencies
RUN apt update -y && apt install -y \
    build-essential \
    wget \
    gpg \
    curl \
    pciutils \
    git \
    cmake \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-opencv \
    libopencv-dev \
    v4l-utils

# Install Python packages
RUN pip install --break-system-packages \
    Flask \
    flask_bootstrap \
    flask-socketio \
    fire \
    psutil \
    zeroconf \
    huggingface_hub

# Install Intel Graphic Drivers
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" > \
    /etc/apt/sources.list.d/intel-gpu-noble.list && \
    apt update -y && \
    apt install -y \
        libze1 \
        intel-level-zero-gpu \
        intel-opencl-icd \
        clinfo \
        libtbb12 

# Install NPU Driver
WORKDIR /tmp
RUN wget -qP /tmp \
        https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-driver-compiler-npu_1.13.0.20250131-13074932693_ubuntu24.04_amd64.deb \
        https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-fw-npu_1.13.0.20250131-13074932693_ubuntu24.04_amd64.deb \
        https://github.com/intel/linux-npu-driver/releases/download/v1.13.0/intel-level-zero-npu_1.13.0.20250131-13074932693_ubuntu24.04_amd64.deb && \
    dpkg -i /tmp/*.deb && rm -f /tmp/*.deb

# Install PCM
RUN git clone --recursive https://github.com/intel/pcm && \
    cd pcm && \
    mkdir build && cd build && \
    cmake .. && cmake --build . --parallel && \
    make install && \
    rm -rf /tmp/pcm


RUN pip install --break-system-packages  -U --pre \
    openvino-genai \
    openvino \
    openvino-tokenizers[transformers] \
        --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly 

RUN pip install --break-system-packages  \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "git+https://github.com/huggingface/optimum-intel.git" \
        "git+https://github.com/openvinotoolkit/nncf.git" \
        "onnx<=1.16.1"

RUN pip install --break-system-packages markdown bleach