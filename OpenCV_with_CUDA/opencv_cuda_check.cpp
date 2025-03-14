/**
 * 测试OpenCV with CUDA加速是否可用
 */ 

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

int main() {
    // 加载CUDA模块
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    // 检查CUDA设备并打印信息
    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    if (deviceCount > 0) {
        std::cout << "OpenCV with CUDA: Yes" << std::endl;
        for (int i = 0; i < deviceCount; ++i) {
            cv::cuda::setDevice(i);
            cv::cuda::printCudaDeviceInfo(i);
        }
    } else {
        std::cout << "OpenCV with CUDA: No" << std::endl;
    }

    return 0;
}
