import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from AlexNet.model import AlexNet

# 定义数据预处理的方法：重置图片大小、变为tensor、归一化
data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor,
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# 加载图片，并增加一个维度
img = Image.open('../12240303_80d87f77a3_n.jpg')
plt.imshow(img)
# 预处理
img = data_transform(img)
# [batch,channel,height,wide]
img = torch.unqueeze(img, dim=0)

# 将事先持久化的clas_index给加载进来
try:
    json_file = open('./class_index.json', 'r')
    class_index = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
# 加载模型
model = AlexNet()
# 读取模型权重
model_weight_path = './weight/AlexNet.pth'
model.load_state_dict(torch.load(model_weight_path))
# 设置为检测模式
model.eval()

# 不需要算梯度
with torch.no_grad():
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
print(class_index[str(predict_cla)], predict[predict_cla].item())
plt.show()