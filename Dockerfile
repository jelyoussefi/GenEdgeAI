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
RUN pip3 install --break-system-packages \
    Flask \
    flask_bootstrap \
    flask-socketio \
    nncf \
    fire \
    psutil \
    openvino-dev[onnx] \
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
RUN wget https://github.com/intel/linux-npu-driver/releases/download/v1.10.0/intel-driver-compiler-npu_1.10.0.20241107-11729849322_ubuntu24.04_amd64.deb \
    https://github.com/intel/linux-npu-driver/releases/download/v1.10.0/intel-fw-npu_1.10.0.20241107-11729849322_ubuntu24.04_amd64.deb \
    https://github.com/intel/linux-npu-driver/releases/download/v1.10.0/intel-level-zero-npu_1.10.0.20241107-11729849322_ubuntu24.04_amd64.deb && \
    dpkg -i *.deb && rm -f *.deb

# Install PCM
RUN git clone --recursive https://github.com/intel/pcm && \
    cd pcm && \
    mkdir build && cd build && \
    cmake .. && cmake --build . --parallel && \
    make install && \
    rm -rf /tmp/pcm

# Install OpenVINO GenAI
RUN pip install --break-system-packages --pre -U openvino-genai --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly 

RUN pip3 install --break-system-packages optimum-intel@git+https://github.com/huggingface/optimum-intel.git
