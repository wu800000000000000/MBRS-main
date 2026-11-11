import cv2
from torch.utils.data import DataLoader
from utils import *
from network.Network import *
from utils.load_test_setting import *
from PIL import Image

# 假设 save_image 函数如下定义
def save_image(image_tensor, save_path):
    # 将 tensor 转换为 PIL Image
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    image_tensor = image_tensor.squeeze()  # 去掉批次维度，使其变为 [3, 100, 100]
    image_tensor = image_tensor * std[:, None, None] + mean[:, None, None]
    image_pil = Image.fromarray(image_tensor.mul(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())

    image_pil.save(save_path)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化网络
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion)
EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
network.load_model_ed(EC_path, map_location=torch.device('cpu'))

# 加载测试数据集
test_dataset = MBRSDataset(os.path.join('D:/MBRS-main/X2', "validation"), H, W)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# 初始化测试结果字典
test_result = {
    "error_rate": 0.0,
    "psnr": 0.0,
    "ssim": 0.0
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

        encoded_images = network.encoder_decoder.module.encoder(images, messages)
        encoded_images = images + (encoded_images - image) * strength_factor

        noised_images = network.encoder_decoder.module.noise([encoded_images, images])

        decoded_messages = network.encoder_decoder.module.decoder(noised_images)


        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        tensor_image = encoded_images.squeeze()  # 去掉批次维度，使其变为 [3, 100, 100]
        tensor_image = tensor_image * std[:, None, None] + mean[:, None, None]

        # psnr
        psnr = - kornia.losses.psnr_loss(encoded_images.detach(), images, 2).item()

        # ssim
        ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

        # 保存编码后的图像
        for j in range(encoded_images.shape[0]):
            save_path = os.path.join(encoded_images_folder, f'encoded_image_{i*batch_size + j}.png')
            save_image(encoded_images[j], save_path)

    error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)

    result = {
        "error_rate": error_rate,
        "psnr": psnr,
        "ssim": ssim,
    }

    for key in result:
        test_result[key] += float(result[key])

    num += 1

    if i in saved_iterations:
        if saved_all is None:
            saved_all = get_random_images(image, encoded_images, noised_images)
        else:
            saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

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
save_images(saved_all, "test", result_folder + "images/", resize_to=(W, H))