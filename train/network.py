import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletNet(nn.Module):
    def __init__(self, embedding_net, device):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.device = device

    def forward(self, x):
        # x1     / x2  / x3
        # anchor / pos / neg:
        #   protein: [B, C, M]
        #   traj:    [B, M, 2]
        x1, x2, x3 = x
        return self.embedding_net(x1.to(self.device)), \
            self.embedding_net(x2.to(self.device)), \
            self.embedding_net(x3.to(self.device))


class TripletLoss(nn.Module):
    def __init__(self, args):
        """论文实现中的联合损失。

        包含两部分:
            1. rank_loss: 约束 anchor-positive 比 anchor-negative 更近
            2. mse_loss : 让嵌入空间距离逼近真实序列距离
        """
        super(TripletLoss, self).__init__()
        self.l, self.r = 1, 1
        step = args.epochs // 5
        self.Ls = {
            step * 0: (10, 0),
            step * 1: (10, 10),
            step * 2: (1, 10),
            step * 3: (0.1, 5),
            step * 4: (0.01, 1),
        }

    def dist(self, ins, pos):
        # 若输入为 [B, T]，直接做向量欧氏距离。
        # 若输入为 [B, C, T]，先在最后一维做每个通道的欧氏距离，再沿通道求和。
        # 这对应论文中“按维度分别计算再求和”的 embedding geometry。
        if len(ins.shape) == 2:
            return torch.norm(ins - pos, dim=1)
        else:
            return torch.sum(torch.norm(ins - pos, dim=-1), dim=-1)

    def forward(self, x, dists, epoch):
        if epoch in self.Ls:
            self.l, self.r = self.Ls[epoch]
        anchor, positive, negative = x
        pos_dist, neg_dist, pos_neg_dist = (d.type(torch.float32) for d in dists)

        # 若 embedding_net 输出 [B, C, T]，则下面三个距离均为 [B]
        pos_embed_dist = self.dist(anchor, positive)
        neg_embed_dist = self.dist(anchor, negative)
        pos_neg_embed_dist = self.dist(positive, negative)

        # 真实距离差值用作 margin，逼近论文里的相对排序约束。
        threshold = neg_dist - pos_dist
        rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)

        mse_loss = (pos_embed_dist - pos_dist) ** 2 + \
                   (neg_embed_dist - neg_dist) ** 2 + \
                   (pos_neg_embed_dist - pos_neg_dist) ** 2

        return torch.mean(self.l * rank_loss), \
            torch.mean(self.r * torch.sqrt(mse_loss)), \
            torch.mean(self.l * rank_loss + self.r * torch.sqrt(mse_loss))
