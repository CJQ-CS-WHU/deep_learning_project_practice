import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from AlexNet.model import AlexNet

# 定义数据预处理的方法：重置图片大小、变为tensor、归一化
data_transform = {
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}
# 加载图片，并增加一个维度
img = Image.open('../12240303_80d87f77a3_n.jpg')
plt.imshow(img)
img = data_transform['test'](img)
# 预处理

# [batch,channel,height,wide]
img = torch.unsqueeze(img, dim=0)

# 将事先持久化的clas_index给加载进来
try:
    json_file = open('./class_index.json', 'r')
    class_index = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)
# 加载模型
model = AlexNet(num_class=5, init_weights=True)
# 读取模型权重
model_weight_path = './weight/AlexNet.pth'
model.load_state_dict(torch.load(model_weight_path))
# 设置为检测模式
model.eval()

# 不需要算梯度
with torch.no_grad():
    # 去掉一个多余的维度（最外）
    output = torch.squeeze(model(img))
    # 用softmax处理成概率，现在是一个数组
    predict = torch.softmax(output, dim=0)
    # 概率最大的一项的index
    predict_cla = torch.argmax(predict).numpy()
# 打印类别名称，打印概率（predict_cla是最大概率对应的index，取predic数组中的对应项就是了）
print('类别：', class_index[str(predict_cla)], '概率：', predict[predict_cla].item())
plt.show()
