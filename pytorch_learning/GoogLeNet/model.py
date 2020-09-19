import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

'''
8/网络框架基本打好，待进一步细化
8/15细化模型细节，但是还需要细化一下
'''

'''
    def __init__(self, in_channels, branch1_1x1_out,
                 branch2_1x1_out, branch2_3x3_out,
                 branch3_1x1_out, branch3_5x5_out,
                 branch4_maxpool_out, branch4_1x1_out):
'''


class GoogLeNet(nn.Module):
    def __init__(self, num_classes, aux_logits, init_weights):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.init_weights = init_weights
        # 输入 layer 1 & layer 2
        self.input_process = InputProcess(3, 192)
        # layer 3
        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer 4
        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layer 5
        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception(832, 384, 293, 384, 48, 128, 128)
        # Auxiliary classifier
        if self.aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes)
            self.aux2 = AuxiliaryClassifier(528, num_classes)
        # layer 6
        self.classifier = Classifier(1024, 5)
        # init_weight
        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        print('input ', x)
        # layer 1 & 2
        x = self.input_process(x)
        print('layer2 ', x)
        # layer 3
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool_3(x)

        # layer 3
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.max_pool_4(x)
        if self.training and self.aux_logits:
            self.aux1 = self.aux1(x)
        # layer 4
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool(x)
        if self.training and self.aux_logits:
            self.aux2 = self.aux2(x)
        # layer 5
        x = self.classifier(x)

        # output
        if self.training and self.aux_logits:
            return x, self.aux1, self.aux2
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


'''
nn.Conv2d(3, 48, kernel_size=11, padding=(1, 2), stride=4),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2),
'''


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding=0, stride=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        print(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class InputProcess(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputProcess, self).__init__()
        self.conv1 = BasicConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # Nx224x224x3
        x = self.conv1(x)
        # Nx112x112x64
        x = self.max_pool1(x)
        # Nx56x56x64
        x = self.conv2(x)
        # Nx56x56x192
        x = self.max_pool2(x)
        # Nx28x28x192
        print(x.shape)
        return x


class Classifier(nn.Module):
    def __init__(self, in_channel, num_class):
        super(Classifier, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.avg=nn.AdaptiveAvgPool2d((1,1))自动根据原有尺寸和目标尺寸计算卷积kernel_size和stride和padding
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(in_channel, num_class)

    def forward(self, x):
        # input Nx7x7x1024
        x = self.avg(x)
        # Nx1x1x1024
        x = torch.flatten(x, 1)
        # Nx1024
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.linear(x)
        # Nx num_classes


'''
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding=0, stride=1):
'''


class Inception(nn.Module):
    def __init__(self, in_channels, branch1_1x1_out,
                 branch2_1x1_out, branch2_3x3_out,
                 branch3_1x1_out, branch3_5x5_out,
                 branch4_1x1_out):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, branch1_1x1_out, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, branch2_1x1_out, kernel_size=1),
            BasicConv2d(branch2_1x1_out, branch2_3x3_out, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, branch3_5x5_out, kernel_size=1),
            BasicConv2d(branch3_1x1_out, branch3_5x5_out, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, branch4_1x1_out, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        # branch,channel,height,width
        output = [branch1, branch2, branch3, branch4]
        return torch.cat(output, 1)


class DeepInception(nn.Module):
    def __init__(self, layer_cfgs):
        super(DeepInception, self).__init__()
        self.layers[len(layer_cfgs)]
        for num, layer_cfg in enumerate(layer_cfgs):
            # 把layer_cfg解析成具体的配置信息
            # 赋值给Sequential
            self.layers[num] = nn.Squential()

    def forward(self, x):
        pass


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_class):
        super(AuxiliaryClassifier, self).__init__()
        self.avg = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 64, kernel_size=1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, num_class)

    def forward(self, x):
        # aux1:Nx512x14x14  aux2:Nx528x14x14
        x = self.avg(x)
        # aux1:Nx512x4x4    aux2:Nx512x4x4
        x = self.conv(x)
        # Nx64x4x4
        x = torch.flatten(x, 1)  # 一共有四个维度（0：branch,1:channels,2:height,3:width），从channel维度开始展平
        # 在输入全连接层前做一个展平，然后做dropout
        x = F.dropout(x, p=0.5, training=self.training)
        # 全连接 Nx1024
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # 全连接 Nx512
        x = self.fc2(x)
        # Nx num_classes
        return x


model = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
print(model)
