from torch import nn

POOL = nn.AvgPool1d
# POOL = nn.AvgPool1d(kernel_size=3, stride=1, padding=1),
ReLU = nn.LeakyReLU(negative_slope=0.01)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 3, 1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        # out = torch.relu(x)
        out = out + x
        return out


class convPretreatment(nn.Module):
    def __init__(self, channel, M, conv_layers=6, conv_channels=10, data_type="protein"):
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
            x = self.MLP(x)
            x = x.permute(0, 2, 1)

        bs = x.shape[0]
        if self.conv_layers == 1:
            x = self.conv1(x)
        elif self.conv_layers != 0:
            x = x.contiguous().view(-1, 1, self.M)
            x = self.conv(x)
            x = x.reshape(bs, self.channel, -1)

        return x


class Pretreatment(nn.Module):
    def __init__(self, pret_type, channel, M, conv_layers, conv_channels, data_type):
        super(Pretreatment, self).__init__()
        self.pret_type = pret_type
        self.channel = channel

        self.convPretreat = convPretreatment(channel=self.channel, M=M, conv_layers=conv_layers,
                                             conv_channels=conv_channels, data_type=data_type)

    def forward(self, x):
        if self.pret_type == "conv":
            x = self.convPretreat(x)

        return x


