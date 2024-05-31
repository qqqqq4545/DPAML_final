import cv2
import numpy as np
import paddlehub as hub
import os

# 加載 deeplabv3p_xception65_humanseg 模型
human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")

def remove_background(frame):
    # 使用人像分割模型进行预测
    result = human_seg.segmentation(images=[frame], visualization=False)
    
    # 获取分割结果
    segmented_image = result[0]['data']
    
    # 创建白色背景
    white_background = np.ones_like(frame) * 255
    
    # 检查分割结果是否有 alpha 通道
    if segmented_image.shape[2] == 4:
        alpha_channel = segmented_image[:, :, 3]  # 获取 alpha 通道
        mask = alpha_channel > 0  # 创建蒙版
        mask = np.stack([mask, mask, mask], axis=-1)  # 转换为三通道蒙版
    else:
        # 如果分割结果没有 alpha 通道，直接使用分割结果作为蒙版
        mask = segmented_image
    
    # 将分割结果作为蒙版应用到白色背景上
    composed_image = np.where(mask, frame, white_background)
    
    return composed_image

def main():
    # 開啟鏡頭
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 去背處理
        result = remove_background(frame)

        # 顯示影像
        cv2.imshow('Original', frame)
        cv2.imshow('Background Removed', result)

        # 按下 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()