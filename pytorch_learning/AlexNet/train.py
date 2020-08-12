import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import torch.optim as optim
import os
import json
from AlexNet.model import AlexNet
import time
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda:0,1' if torch.cuda.is_available() else 'cpu')
print(device)

# 数据预处理字典
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}
# 定义数据集路径，并将数据文件读入到train_dataset中。
data_root = os.path.abspath(os.path.join(os.getcwd(), '..\\'))
image_path = data_root + "\\data_sets\\flower_data\\"
# 用imagefolder读取训练集，并根据transform做数据预处理
train_dataset = datasets.ImageFolder(root=image_path + '\\train',
                                     transform=data_transform['train'])
# 用imagefolder读取验证集，并根据transform做数据预处理
validate_dataset = datasets.ImageFolder(root=image_path + '\\val',
                                        transform=data_transform['test'])
# 计算训练集大小，以备以后使用
train_num = len(train_dataset)
# 计算验证机的大小，以备以后使用
val_num = len(validate_dataset)
# 得到一个字典，包含每个类别所对应的索引值
flower_list = train_dataset.class_to_idx
print(flower_list)
# 将标签字典的键和值对调，以备根据index来查询类别名称
cla_dict = dict((val, key) for key, val in flower_list.items())
print(cla_dict)
# 转为json
json_str = json.dumps(cla_dict, indent=4)
# 存到json文件
with open('class_index.json', 'w') as json_file:
    json_file.write(json_str)
# 加载数据
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# test_data_iter = iter(val_loader)
# test_image, test_label = test_data_iter.next()

# print(cla_dict[test_label[1].item()])
#
#
# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))

net = AlexNet(num_class=5, init_weights=True)
net.to(device)
print("网络结构", net)
loss_function = nn.CrossEntropyLoss()
print("损失函数", loss_function)
optimizer = optim.Adam(net.parameters(), lr=0.0002)
print("优化器", optimizer)
save_path = './AlexNet.pth'
best_acc = 0.0
for epoch in range(1):
    # 切换到训练状态（dropout启动）
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        rate = (step + 1) / len(train_loader)
        a = '*' * int(rate * 50)
        b = '.' * int((1 - rate) * 50)
        print("\rtrain loss:{:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print('cost time:', time.perf_counter() - t1)

    net.eval()
    acc = 0.0
    save_path = './weight/AlexNet.pth'
    with torch.no_grad():
        for data_test in val_loader:
            test_images, test_labels = data_test
            outputs = net(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
        accurate_test = acc / val_num
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f test_accuracy: %.3f'
              % (epoch + 1, running_loss / step, acc / val_num))
