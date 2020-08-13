import torch
import torch.nn as nn
import torchvision

'''
网络框架基本打好，待进一步细化
'''


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        # layer 1
        self.input_process = InputProcess()
        # layer 2
        self.inception_3a = Inception()
        self.inception_3b = Inception()
        self.max_pool_3 = nn.MaxPool2d()
        # layer 3
        self.inception_4a = Inception()
        self.inception_4b = Inception()
        self.inception_4c = Inception()
        self.inception_4d = Inception()
        self.inception_4e = Inception()
        self.max_pool_4 = nn.MaxPool2d()
        # layer 4
        self.inception_5a = Inception()
        self.inception_5b = Inception()
        self.avg_pool = nn.AvgPool2d()
        # layer 5
        self.classifier = Classifier()
        # Auxiliary classifier
        self.aux1 = AuxiliaryClassifier()
        self.aux2 = AuxiliaryClassifier()

    def forward(self, x):
        # layer 1
        x = self.input_process(x)
        # layer 2
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.max_pool_3(x)
        if self.training:
            aux1 = self.aux1(x)
        # layer 3
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.max_pool_4(x)
        if self.training:
            aux2 = self.aux2(x)
        # layer 4
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool(x)
        # layer 5
        x = self.classifier(x)

        # output
        if self.training:
            return [x, aux1, aux2]
        return x


'''
nn.Conv2d(3, 48, kernel_size=11, padding=(1, 2), stride=4),
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2),
'''


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, stride):
        super(BasicConv2d, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.layer(x)


class InputProcess(nn.Module):
    def __init__(self):
        super(InputProcess, self).__init__()
        self.layer1 = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, padding=5, stride=2),
            nn.MaxPool2d()
        )
        self.layer2 = nn.Sequential(
            BasicConv2d(),
            nn.MaxPool2d()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layer(x)


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d()
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(),
            BasicConv2d()
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(),
            BasicConv2d()
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(),
            BasicConv2d()
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return [x1, x2, x3, x4]


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
    def __init__(self):
        super(AuxiliaryClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.AvgPool1d(),
            BasicConv2d(),
            nn.Linear(),
            nn.Linear(),
            nn.Softmax()
        )

    def forward(self, x):
        return self.layer(x)
