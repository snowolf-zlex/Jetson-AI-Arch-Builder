#!/bin/bash

# 编译命令
g++ -o opencv_cuda_check opencv_cuda_check.cpp \
    `pkg-config --cflags --libs opencv4` \
    -I/usr/include/opencv4 \
    -I/usr/local/include \
    -L/usr/local/cuda/lib64 \
    -lopencv_core -lopencv_highgui -lopencv_cudaarithm -lopencv_cudaimgproc

# 运行编译后的程序
./opencv_cuda_check