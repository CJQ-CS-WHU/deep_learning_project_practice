from abc import ABC

import torch
import torch.nn as nn

# 网络结构的配置字典
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 'M']
}


# 根据配置字典，构建序列化的特征提取模型
def make_features(cfg: list):
    layers = []
    in_channels = 3
    for kernel_num in cfg:
        if kernel_num == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, kernel_num, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = kernel_num
    # 将列表通过非关键字参数的形式传入（*）
    return nn.Sequential(*layers)


# 根据模型名称来实例化一个VGG网络
def vgg(model_name='vgg16', **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print('Warning: model {} not in cfgs dict').format(model_name)
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model


# VGG类
class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        # 声明特征提取网络
        self.features = features
        # 声明分类器（假设已把input展平）
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, class_num)
        )
        # 如果init_weights为真，则要对所有模块进行初始化
        if init_weights:
            self._init_weights()
        pass

    def forward(self, x):
        # Nx3x224x224
        x = self.features(x)
        # Nx512x7x7
        x = torch.flatten(x, start_dim=1)
        # Nx512*7*7
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


model = vgg('vgg16')
print(model)
