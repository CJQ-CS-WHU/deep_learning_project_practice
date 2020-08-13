import torch
from torch import optim
from torch import nn
from torchvision import transforms
from torchvision import datasets
import os
import time

import json
from VGG import model

# 选定设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
# 读取数据
#   路径
data_root = os.path.abspath(os.path.join(os.getcwd(), '..\\'))
image_path = data_root + '\\data_sets\\flower_data'
#   读取
train_dataset = datasets.ImageFolder(root=image_path+'\\train',
                                     target_transform=data_transform['train'])

test_dataset = datasets.ImageFolder(root=image_path+'\\val',
                                    target_transform=data_transform['test'])

num_tarin = len(train_dataset)
print(num_tarin)
num_val = len(test_dataset)
print(num_val)

#   类标签
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)

with open('class_index.json', 'w') as json_file:
    json_file.write(json_str)

# 加载数据
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# 实例化网络，定义损失函数、优化器
net = model.vgg('vgg16')
print('网络结构：', net)
loss_func = nn.CrossEntropyLoss()
print('损失函数：', loss_func)
optimizer = optim.Adam(net.parameters(), lr=0.001)
print('优化器：', optimizer)

# 多轮训练、打印过程和测试结果、保存模型
for epoch in range(1):
    running_loss = 0
    net.train()
    it=iter(train_loader)
    print(list(enumerate(train_loader)))
    for step, data in enumerate(train_loader, start=0):
        p1 = time.perf_counter()
        images, label = data
        optimizer.zero_grad()
        output = net(images)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss
        rate = (step + 1) / len(train_loader)
        a = '*' * rate * 50
        b = '.' * rate * 50
        print('\repoch:{}:{:.3.0f}%[{}->{}] train loss:{:.3f}'.format(epoch, int(rate * 100), a, b, running_loss),
              end="")
    print()
    print('cost time:{}', float(time.perf_counter() - p1))
    net.eval()
    acc = 0.0
    best_acc = 0
    save_path = './weight/vgg.pth'
    with torch.no_grad():
        for data_test in val_loader():
            test_images, test_labels = data_test
            outputs = net(test_images)
            predicts = torch.max(outputs, dim=1)[1]
            acc += (predicts == test_labels).sum().item()
        accurate_test = acc / num_val
        if accurate_test > best_acc:
            best_acc = accurate_test
            torch.save(net.state_dict(), save_path)
        print('test accurate:{}'.format(best_acc))
