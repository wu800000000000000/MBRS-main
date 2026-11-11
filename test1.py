from torch.utils.data import DataLoader
from utils import *
from network.Network import *
from utils.load_test_setting import *
from PIL import Image

# 假设 save_image 函数如下定义
def save_image(image_tensor, save_path):
    # 将 tensor 转换为 PIL Image
    image_pil = Image.fromarray(image_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
    image_pil.save(save_path)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion)
EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
network.load_model_ed(EC_path, map_location=torch.device('cpu'))

# 加载测试数据集
test_dataset = MBRSDataset(os.path.join('D:/MBRS-main/X2', "test2"), H, W)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) # 创建了一个用于测试的数据加载器，它将从 test_dataset 中加载数据

# 初始化测试结果字典
test_result = {
    "error_rate": 0.0,
    #"psnr": 0.0,
    #"ssim": 0.0
}

# 记录开始时间
start_time = time.time()

# 随机选择要保存的迭代次数
saved_iterations = np.random.choice(np.arange(len(test_dataset)), size=save_images_number, replace=False)
saved_all = None

# 确保保存图像的目录存在
encoded_images_folder = os.path.join(result_folder, "encoded_images")
os.makedirs(encoded_images_folder, exist_ok=True)

# 测试循环
num = 0
for i, images in enumerate(test_dataloader):
    image = images.to(device)
    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

    network.encoder_decoder.eval()
    network.discriminator.eval()

    with torch.no_grad():
        # use device to compute
        images, messages = images.to(network.device), message.to(network.device)

        # 直接使用原始图像进行解码
        decoded_messages = network.encoder_decoder.module.decoder(images)

        # psnr
        psnr = kornia.losses.psnr_loss(images.detach(), images, 2).item()

        # ssim
        ssim = 1 - 2 * kornia.losses.ssim_loss(images.detach(), images, window_size=5, reduction="mean")

        # 保存原始图像
        # for j in range(images.shape[0]):
            #save_path = os.path.join(encoded_images_folder, f"original_image_{i*batch_size + j}.png")
            #save_image(images[j], save_path)

    error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)
    #
    result = {
        "error_rate": error_rate,
        #"psnr": psnr,
        #"ssim": ssim,
    }

    for key in result:
        test_result[key] += float(result[key])

    num += 1

    if i in saved_iterations:
        if saved_all is None:
            saved_all = get_random_images(image, images, images)  # 使用原始图像
        else:
            saved_all = concatenate_images(saved_all, image, images, images)

    content = "Image " + str(i) + " : \n"
    for key in test_result:
        content += key + "=" + str(result[key]) + ","
    content += "\n"

    with open(test_log, "a") as file:
        file.write(content)

    print(content)

# 计算平均值并保存
content = "Average : \n"
for key in test_result:
    content += key + "=" + str(test_result[key] / num) + ","
content += "\n"

with open(test_log, "a") as file:
    file.write(content)

print(content)

# 保存图像
# save_images(saved_all, "test", result_folder + "images/", resize_to=(W, H))