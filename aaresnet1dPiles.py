import torch.nn as nn
import warnings
import matplotlib.pyplot as plt
import torch

torch.cuda.init()
cuda = torch.device('cuda')
warnings.filterwarnings("ignore")

plt.ion()

##ResNeXt 基本块（ResNet的变种）（https://zhuanlan.zhihu.com/p/51075096）（https://github.com/WZMIAOMIAO/deep-learning-for-image-processing）
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=4):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False).to(cuda)  # squeeze channels
        self.bn1 = nn.BatchNorm1d(width).to(cuda)
        # -----------------------------------------
        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1).to(cuda)
        self.bn2 = nn.BatchNorm1d(width).to(cuda)
        # -----------------------------------------
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False).to(cuda)  # unsqueeze channels
        self.bn3 = nn.BatchNorm1d(out_channel*self.expansion).to(cuda)
        self.relu = nn.ReLU(inplace=True).to(cuda)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


##自适应注意力机制（Adaptive Attention Mechanism Module）（https://blog.csdn.net/qq_40379132/article/details/125127135）
# CBMA  通道注意力机制和空间注意力机制的结合
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1).to(cuda)  # 平均池化高宽为1
        self.max_pool = nn.AdaptiveMaxPool1d(1).to(cuda)  # 最大池化高宽为1

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False).to(cuda)
        self.relu1 = nn.ReLU().to(cuda)
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False).to(cuda)

        self.bn1 = nn.BatchNorm1d(in_planes).to(cuda)

        self.sigmoid = nn.Sigmoid().to(cuda)

    def forward(self, x):
        # 平均池化---》1*1卷积层降维----》激活函数----》卷积层升维
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化---》1*1卷积层降维----》激活函数----》卷积层升维
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # 加和操作
        out = self.bn1(out)
        return self.sigmoid(out)  # sigmoid激活操作


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        # 经过一个卷积层，输入维度是2，输出维度是1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False).to(cuda)
        self.bn1 = nn.BatchNorm1d(1).to(cuda)
        self.sigmoid = nn.Sigmoid().to(cuda)  # sigmoid激活操作

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w
        x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
        x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
        return self.sigmoid(x)  # sigmoid激活操作


class cbamblock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbamblock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
        x = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层
        return x


##Automatic Modulation Recognition Based on Adaptive Attention Mechanism and ResNeXt WSL Model
##就叫做AAResNet吧
## 时频图 3*288*288
#from torchsummary import summary


class AAResNet(nn.Module):
    def __init__(self,
                 block=Bottleneck,
                 blocks_num=[3, 4, 6, 3],
                 num_classes=35,
                 include_top=True,
                 groups=32,
                 in_channel=2,
                 width_per_group=4,
                 return_feature=False):
        super(AAResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.return_feature = return_feature

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv1d(in_channel, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False).to(cuda)
        self.bn1 = nn.BatchNorm1d(self.in_channel).to(cuda)
        self.relu = nn.ReLU(inplace=True).to(cuda)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1).to(cuda)

        self.layer1 = self._make_layer(block, 64, blocks_num[0]).to(cuda)
        self.attention1 = cbamblock(256)

        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2).to(cuda)
        self.attention2 = cbamblock(512)

        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2).to(cuda)
        self.attention3 = cbamblock(1024)

        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2).to(cuda)
        self.attention4 = cbamblock(2048)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1).to(cuda)  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes).to(cuda)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channel * block.expansion)).to(cuda)

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.attention1(x)

        x = self.layer2(x)
        # x = self.attention2(x)

        x = self.layer3(x)
        # x = self.attention3(x)

        x = self.layer4(x)
        # x = self.attention4(x)

        if self.include_top:
            x = self.avgpool(x)
            feature = torch.flatten(x, 1)
            if self.return_feature:
                return feature
            x = self.fc(feature)
        return x

# 
# if __name__ == '__main__':
#     model = AAResNet()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     summary(model, input_size=(2, 256))
