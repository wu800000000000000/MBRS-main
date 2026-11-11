import numpy as np
import cv2   # 导入OpenCV库

# 读取图像
image = cv2.imread(r'C:\Users\86181\Desktop\MBRS-main\X2\test4\20444c4f2f08e2963499e0efa02d20f.jpg')

# 检查图像是否正确加载
if image is None:
    print("Error: Image not found.")
    exit()

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊
blurred = cv2.GaussianBlur(gray, (5, 5), 0) # 对灰度图像应用高斯模糊，以减少图像噪声和细节

# 应用阈值操作，将图像转换为二值图像
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 寻找轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓
if contours:
    max_contour = max(contours, key=cv2.contourArea) # 如果存在轮廓，找到面积最大的轮廓。

    # 获取最小外接矩形
    rect = cv2.minAreaRect(max_contour) # 对于最大的轮廓，找到最小外接矩形
    box = cv2.boxPoints(rect)  # 将矩形转换为四个角点
    box = np.int32(box)  # 转换为整数

    # 对角点进行排序，确保裁剪区域正确
    # 找到左上角和右下角的点
    x_coords = box[:, 0]
    y_coords = box[:, 1]
    top_left = box[np.argmin(y_coords)]
    bottom_right = box[np.argmax(y_coords)]

    # 裁剪矩形区域
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # 检查裁剪后的图像是否为空
    if cropped_image.size > 0:
        # 保存裁剪后的图像
        cv2.imwrite(r'C:\Users\86181\Desktop\MBRS-main\X2\test4_colored_square.jpg', cropped_image)
    else:
        print("The cropped image is empty.")
else:
    print("No contours found.")