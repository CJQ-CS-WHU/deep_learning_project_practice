import torch

# 选定设备
device = torch.device('cuda:0,1,2,3,4,5,6,7' if torch.cuda.is_available() else 'cpu')
print(device)

data_transform = {
}
