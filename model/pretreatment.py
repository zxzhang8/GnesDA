from torch import nn

POOL = nn.AvgPool1d
# POOL = nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
ReLU = nn.LeakyReLU(negative_slope=0.01)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """一维残差块，作用在单个序列通道上。"""
        super(ResBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 3, 1, padding=1, bias=False)

    def forward(self, x):
        # x/out: [B, C, L]
        out = self.conv(x)
        # out = torch.relu(x)
        out = out + x
        return out


class convPretreatment(nn.Module):
    def __init__(self, channel, M, conv_layers=6, conv_channels=10, data_type="protein"):
        """论文中的 sequence-specific transformation + convolutional block 前半段。

        两类输入的处理方式:
            - protein:
                输入已经是 one-hot 矩阵 [B, C, M]
            - traj:
                输入为原始二维轨迹 [B, M, 2]
                先通过 MLP 将每个点从 2 维映射到 channel 维，
                再转成 [B, channel, M] 交给后续卷积块
        """
        super(convPretreatment, self).__init__()
        self.data_type = data_type
        self.channel = channel
        self.M = M
        self.conv_layers = conv_layers
        self.MLP = nn.Sequential(
            nn.Linear(2, channel),
            nn.ReLU(),
            nn.Linear(channel, channel),
        )
        self.conv1 = nn.Sequential(
                ResBlock(channel, channel),
                POOL(2),
                nn.ReLU()
            )    
        if conv_layers != 0:
            self.conv = nn.Sequential(
                nn.Conv1d(1, conv_channels, 3, 1, padding=1, bias=False),
                POOL(2),
            )
            for i in range(conv_layers - 2):
                self.conv.add_module("res{}".format(i + 1), ResBlock(conv_channels, conv_channels))
                self.conv.add_module("pool{}".format(i + 1), POOL(2))
                if (i + 2) % 3 == 0:
                    self.conv.add_module("relu{}".format(0), nn.ReLU())
            self.conv.add_module("conv{}".format(conv_layers - 1), nn.Conv1d(conv_channels, 1, 3, 1, padding=1, bias=False))
            self.conv.add_module("pool{}".format(conv_layers - 1), POOL(2))
            self.conv.add_module("relu{}".format(0), nn.ReLU())

    def forward(self, x):
        if self.data_type == "traj":
            # x: [B, M, 2]
            # 将每个轨迹点的二维坐标非线性映射到统一通道数 channel
            x = self.MLP(x)
            # [B, M, channel] -> [B, channel, M]
            x = x.permute(0, 2, 1)

        bs = x.shape[0]
        if self.conv_layers == 1:
            # x: [B, channel, M] -> [B, channel, M/2]
            x = self.conv1(x)
        elif self.conv_layers != 0:
            # 这里将每个通道视为一个独立的一维序列:
            # [B, channel, M] -> [B * channel, 1, M]
            x = x.contiguous().view(-1, 1, self.M)
            # 经过多层卷积/池化后输出 [B * channel, 1, M_conv]
            x = self.conv(x)
            # 恢复回按通道组织的表示: [B, channel, M_conv]
            x = x.reshape(bs, self.channel, -1)

        return x


class Pretreatment(nn.Module):
    def __init__(self, pret_type, channel, M, conv_layers, conv_channels, data_type):
        """输入预处理封装层。"""
        super(Pretreatment, self).__init__()
        self.pret_type = pret_type
        self.channel = channel

        self.convPretreat = convPretreatment(channel=self.channel, M=M, conv_layers=conv_layers,
                                             conv_channels=conv_channels, data_type=data_type)

    def forward(self, x):
        if self.pret_type == "conv":
            # 输出统一为 [B, channel, L_conv]
            x = self.convPretreat(x)

        return x

