import cv2 # 导入cv2（OpenCV库）用于图像处理
import numpy as np
from imutils.perspective import four_point_transform # imutils.perspective中的four_point_transform用于图像的四点透视变换。

# 读取图像
image = cv2.imread('D:/MBRS-main/X2/test4/Figure_4.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 使用cv2.imread读取指定路径的图像，并将其转换为灰度图像。
retval, thread_img = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV) # 对灰度图像进行阈值处理(全局阈值处理算法)，得到二值图像。

# 获得轮廓列表
contours, hierarchy = cv2.findContours(thread_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # 使用cv2.findContours找到二值图像中的轮廓
# 最大轮廓
largest_contour = max(contours, key=cv2.contourArea) # 从轮廓列表中找到面积最大的轮廓。

docCnt = None  # 这个变量将用来存储检测到的四边形轮廓
epsilon = 0.04 * cv2.arcLength(largest_contour, True)  # 轮廓长度
approx = cv2.approxPolyDP(largest_contour, epsilon, True)  # 获取近似的轮廓
if len(approx) == 4:  # 近似轮廓有四个顶点
    docCnt = approx

result_img = four_point_transform(image, docCnt.reshape(4, 2))  # 对原始图像进行四点透视变换,通常用于矫正图像中的透视失真。

cv2.namedWindow('img', 0)
cv2.imshow('img', image)
cv2.namedWindow('result_img', 0)
cv2.imshow('result_img', result_img)

# 保存裁剪后的图片
cv2.imwrite('D:/MBRS-main/X2/test4/3.png', result_img)

cv2.waitKey(0)  # cv2.waitKey(0)让程序等待用户按键，
cv2.destroyAllWindows()  #  cv2.destroyAllWindows()则关闭所有OpenCV创建的窗口