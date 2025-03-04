#!/bin/bash

# ----------------------------
# Jetson AI框架初始化 
# 基于Ubuntu 22.04 LTS 
# V1.0
# By Snowolf
# 2025-02-06

# 说明：
# 将执行以下动作：
# 1. Torch
# 2. TorchVision
# 3. TorchAudio
# 4. TensorFlow
# ----------------------------

# 设置错误退出
set -e

# ============================
# 预备工作: 安装必要的依赖和工具
# ============================

install_dependencies() {
    echo "===== 安装 Ninja 和编译工具 ====="
    sudo apt update && apt-get install -y \
        ninja-build \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
        libpython3-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libopenblas-dev \
        libomp-dev
}

# ============================
# Torch 安装
# ============================

install_pytorch() {
    echo "===== 安装 PyTorch ====="
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install libcusparselt0 libcusparselt-dev
    # 下载并安装 PyTorch 的 Wheel 文件
    wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
    pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
}

# ============================
# TorchVision 安装
# ============================

install_torchvision() {
    echo "===== 安装 TorchVision ====="
    # 克隆并安装 torchvision
    git clone --branch v0.20.0 https://github.com/pytorch/vision torchvision
    cd torchvision
    export BUILD_VERSION=0.20.0
    python3 setup.py clean
    python3 setup.py bdist_wheel
    pip install dist/torchvision*.whl
    cd ..
}

# ============================
# TorchAudio 安装
# ============================

install_torchaudio() {
    echo "===== 安装 TorchAudio ====="
    # 克隆并安装 torchaudio
    git clone --branch v2.6.0 https://github.com/pytorch/audio.git torchaudio
    cd torchaudio
    export BUILD_VERSION=2.6.0
    python3 setup.py clean
    python3 setup.py bdist_wheel
    pip install dist/torchaudio*.whl
    cd ..
}

# ============================
# TensorFlow 安装
# ============================

install_tensorflow() {
    echo "===== 安装 TensorFlow ====="
    # 下载并安装 TensorFlow 的 Wheel 文件
    wget https://developer.download.nvidia.com/compute/redist/jp/v61/tensorflow/tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl
    pip install tensorflow-2.16.1+nv24.08-cp310-cp310-linux_aarch64.whl
}

# ============================
# 验证 GPU 是否可用
# ============================

verify_gpu() {
    echo "===== 验证 GPU 是否可用 ====="
    python3 -c "import torch torchvision torchaudio; print('CUDA Available:', torch.cuda.is_available())"
    python3 -c "import tensorflow as tf; print('TensorFlow GPU Available:', tf.test.is_gpu_available())"
}

# ============================
# 执行安装和验证
# ============================

echo "===== Jetson AI框架初始化 ====="
sleep 1

# 执行所有步骤
install_dependencies
install_pytorch
install_torchvision
install_torchaudio
install_tensorflow

# 验证 GPU 是否可用
verify_gpu

echo "安装完成！"
