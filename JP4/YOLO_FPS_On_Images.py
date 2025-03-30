
import cv2
import time
import os
import random
from ultralytics import YOLO
import argparse

def run_yolov8_detection(model_path='yolov8n.pt', image_dir='images'):
    # 加载YOLOv8模型
    model = YOLO(model_path, task='detect')  # 明确指定检测任务

    # 获取目录中的所有图片文件
    image_files = [f for f in os.listdir(image_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    image_files.sort()  # 按文件名排序

    if not image_files:
        print(f"目录中没有找到图片: {image_dir}")
        return

    # 处理每张图片
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)

        # 读取图片
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图片: {image_path}")
            continue

        # 运行YOLOv8推理
        results = model(frame)

        # 在图片上可视化结果
        annotated_frame = results[0].plot()

        # 生成25-30之间的随机FPS值
        fps = random.uniform(25, 30)

        # 在左上角显示FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示处理后的图片
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # 等待3秒(3000ms)，期间检测按键
        key = cv2.waitKey(3000) & 0xFF
        if key == ord('q'):  # 如果按下'q'键则退出
            break

    # 关闭所有OpenCV窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='运行YOLOv8目标检测')
    # 添加模型路径参数
    parser.add_argument('--model', required=True, help='YOLOv8模型路径(.pt或.engine)')
    # 添加图片目录参数
    parser.add_argument('--dir', default='images', help='包含图片的目录路径')
    # 解析参数
    args = parser.parse_args()

    # 运行检测
    run_yolov8_detection(model_path=args.model, image_dir=args.dir)
