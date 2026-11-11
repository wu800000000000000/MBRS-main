from torch.utils.data import DataLoader
from utils import *
from network.Network import *

from utils.load_train_setting import *

'''
train
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, only_decoder)

train_dataset = MBRSDataset(os.path.join('D:/MBRS-main/X2',"train" ), H, W)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = MBRSDataset(os.path.join('D:/MBRS-main/X2', "validation"), H, W)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

if train_continue: # 如果train_continue为真,EC_path和D_path这两个变量分别用于存储编码器（EC）和解码器（D）的模型权重路径
	EC_path = "results/" + train_continue_path + "/models/EC_" + str(train_continue_epoch) + ".pth"
	D_path = "results/" + train_continue_path + "/models/D_" + str(train_continue_epoch) + ".pth"
	network.load_model(EC_path, D_path)

print("\nStart training : \n\n")

for epoch in range(epoch_number): # 如果train_continue为真，则epoch加上train_continue_epoch的值；否则epoch加上0

	epoch += train_continue_epoch if train_continue else 0

	running_result = {
		"error_rate": 0.0, # 错误率
		"psnr": 0.0,
		"ssim": 0.0,
		"g_loss": 0.0,  # 生成器总损失
		"g_loss_on_discriminator": 0.0, # 生成器在判别器上的二元交叉熵损失
		"g_loss_on_encoder": 0.0, # 原始图像与编码图像的均方误差损失函数
		"g_loss_on_decoder": 0.0, # 解码信息与原始信息的均方误差损失函数
		"d_cover_loss": 0.0,  # 判别器覆盖损失，判断他是覆盖图片的概率损失1
		"d_encoded_loss": 0.0, # 判别器编码损失，判断他是编码图片的概率损失0
		# ↓↓↓ 新增这两个时间字段 ↓↓↓
		"encoding_time": 0.0,
		"decoding_time": 0.0
	}

	start_time = time.time()

	'''
	train        
	'''
	num = 0
	for _, images, in enumerate(train_dataloader):  # 遍历训练数据加载器
		image = images.to(device)  # 将图像数据移动到指定的设备（如GPU）
		message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device) # 生成一个随机的0和1的张量作为消息，并移动到指定设备

		result = network.train(image, message) if not only_decoder else network.train_only_decoder(image, message) # 如果only_decoder不为真，则使用image和message对网络进行训练；否则仅使用解码器对网络进行训练

		for key in result:  # 遍历训练结果中的每个键
			running_result[key] += float(result[key])  # 将结果累加到running_result字典中

		num += 1   # 增加迭代次数

	'''
	train results 训练结果     # 这段代码的功能是将模型训练过程中的结果记录到文件并打印到控制台
	'''
	content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"  # 构建一个包含当前轮次信息和运行结果的字符串epoch
	for key in running_result:
		content += key + "=" + str(running_result[key] / num) + ","   # 以及每个运行结果（running_result 字典中的键值对）的平均值
	content += "\n"

	with open(result_folder + "/train_log.txt", "a") as file:  # 将这个字符串追加写入到 train_log.txt 文件中
		file.write(content)
	print(content)  # 在控制台打印这个字符串

	'''
	validation 验证
	'''

	val_result = {
		"error_rate": 0.0,
		"psnr": 0.0,
		"ssim": 0.0,
		"g_loss": 0.0,
		"g_loss_on_discriminator": 0.0,
		"g_loss_on_encoder": 0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0,
		# ↓↓↓ 新增这两个时间字段 ↓↓↓
		"encoding_time": 0.0,
		"decoding_time": 0.0
	}

	start_time = time.time()

	saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False) # 从验证数据加载器val_dataloader中随机选择save_images_number个不重复的样本索引
	saved_all = None  # 初始化saved_all变量，用于后续可能的操作，比如存储这些样本的数据

	num = 0
	for i, images in enumerate(val_dataloader):  # 遍历验证数据加载器 val_dataloader 中的每个批次的图像
		image = images.to(device)   # 将图像数据移动到指定的设备（如 GPU）上
		message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
# 调用network的validation方法，传入图像和消息，获取验证结果和一系列中间输出。
		result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message)

		for key in result:  # 遍历result字典中的每个键
			val_result[key] += float(result[key])  # 将result字典中每个键对应的值累加到 val_result 字典中对应的键上

		num += 1

		if i in saved_iterations: # 检查当前迭代索引i是否在之前随机选择的saved_iterations中
			if saved_all is None: # 如果saved_all是None（即还没有保存任何图像）则调用get_random_images函数来获取随机图像，并将结果赋值给saved_all
				saved_all = get_random_images(image, encoded_images, noised_images)
			else:  # 否则调用concatenate_images函数来将当前的图像与之前保存的图像拼接起来，并更新saved_all的值
				saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

	save_images(saved_all, epoch, result_folder + "images/", resize_to=(W, H)) # saved_all中的图像保存到文件系统中，保存的路径是result_folder下的 "images/"目录，并且将图像大小调整为 (W, H)

	'''
	validation results 验证结果   
	'''
	content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
	for key in val_result:
		content += key + "=" + str(val_result[key] / num) + ","
	content += "\n"

	with open(result_folder + "/val_log.txt", "a") as file:
		file.write(content)
	print(content)

	'''
	save model 保存模型
	'''
	path_model = result_folder + "models/"
	path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
	path_discriminator = path_model + "D_" + str(epoch) + ".pth"
	network.save_model(path_encoder_decoder, path_discriminator)


def count_parameters(model):
	"""统计模型可训练参数总数"""
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 分别统计编码器、解码器、判别器的参数量
encoder_param = count_parameters(network.encoder) if hasattr(network, 'encoder') else 0
decoder_param = count_parameters(network.decoder) if hasattr(network, 'decoder') else 0
discriminator_param = count_parameters(network.discriminator) if hasattr(network, 'discriminator') else 0
total_param = encoder_param + decoder_param + discriminator_param

# 构建参数量信息
param_info = (
	f"Model Parameters:\n"
	f"Encoding={encoder_param}, Decoding={decoder_param}, "
	f"Discriminator={discriminator_param}, Total={total_param}\n"
)

# 打印并写入日志
print(param_info)
with open(result_folder + "/train_log.txt", "a") as f:
	f.write(param_info)