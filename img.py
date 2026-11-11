import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread('D:/MBRS-main/X2/test4/2.jpg', cv2.IMREAD_GRAYSCALE)

# 应用Canny边缘检测，调整阈值以提高精细度
edges = cv2.Canny(image, 50, 150)  # 降低阈值

# 显示原始图像和边缘检测结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# 保存边缘检测结果
cv2.imwrite('D:/MBRS-main/X2/test4/4.jpg', edges)