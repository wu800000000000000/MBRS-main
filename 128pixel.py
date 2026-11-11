import cv2
import os
import glob

# 设置图片的宽度和高度
width = 128
height = 128

# 设置输入和输出文件夹的路径
input_folder = r'C:\Users\86181\Desktop\MBRS-main\X2\test1'  # 使用原始字符串避免转义问题
output_folder = r'C:\Users\86181\Desktop\MBRS-main\X2\test2'  # 使用原始字符串避免转义问题

# 如果输出文件夹不存在，则创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取输入文件夹中所有的PNG图片文件
images = glob.glob(os.path.join(input_folder, '*.jpg'))  # 只匹配PNG文件

# 遍历每张图片
for image_path in images:
    try:
        # 读取图片
        img = cv2.imread(image_path)

        # 确保图片被正确读取
        if img is not None:
            # 缩放图片
            resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA) # 区域插值法

            # 设置输出图片的完整路径
            filename = os.path.basename(image_path)  # 获取文件名
            output_path = os.path.join(output_folder, filename)

            # 保存缩放后的图片
            cv2.imwrite(output_path, resized_img)
        else:
            print(f"图片 {image_path} 无法读取，可能是因为文件不是有效的图片格式。")
    except Exception as e:
        print(f"处理图片 {image_path} 时发生错误：{e}")

print("所有可读的PNG图片已缩放到128x128像素，并保存在指定的输出文件夹中。")