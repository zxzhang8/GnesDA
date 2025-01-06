import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletNet(nn.Module):
    def __init__(self, embedding_net, device):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
        self.device = device

    def forward(self, x):
        x1, x2, x3 = x
        return self.embedding_net(x1.to(self.device)), \
            self.embedding_net(x2.to(self.device)), \
            self.embedding_net(x3.to(self.device))


class TripletLoss(nn.Module):
    def __init__(self, args):
        super(TripletLoss, self).__init__()
        self.l, self.r = 1, 1
        step = args.epochs // 5
        self.Ls = {
            step * 0: (0, 10),
            step * 1: (10, 10),
            step * 2: (10, 1),
            step * 3: (5, 0.1),
            step * 4: (1, 0.01),
        }

    def dist(self, ins, pos):
        if len(ins.shape) == 2:
            return torch.norm(ins - pos, dim=1)
        else:
            return torch.sum(torch.norm(ins - pos, dim=-1), dim=-1)

    def forward(self, x, dists, epoch):
        if epoch in self.Ls:
            self.l, self.r = self.Ls[epoch]
        anchor, positive, negative = x
        pos_dist, neg_dist, pos_neg_dist = (d.type(torch.float32) for d in dists)

        pos_embed_dist = self.dist(anchor, positive)
        neg_embed_dist = self.dist(anchor, negative)
        pos_neg_embed_dist = self.dist(positive, negative)

        threshold = neg_dist - pos_dist
        rank_loss = F.relu(pos_embed_dist - neg_embed_dist + threshold)

        mse_loss = (pos_embed_dist - pos_dist) ** 2 + \
                   (neg_embed_dist - neg_dist) ** 2 + \
                   (pos_neg_embed_dist - pos_neg_dist) ** 2

        if epoch == -1:
            print("pos_embed_dist:{}".format(pos_embed_dist))
            print("neg_embed_dist:{}".format(neg_embed_dist))
            print("pos_neg_embed_dist:{}".format(pos_neg_embed_dist))
            print("pos_dist:{}".format(pos_dist))
            print("neg_dist:{}".format(neg_dist))
            print("pos_neg_dist:{}".format(pos_neg_dist))
            print("rank_loss:{}".format(rank_loss))
            print("mse_loss:{}".format(mse_loss))
            for i in range(32):
                print(anchor[i])
                print(positive[i])
                print(negative[i])

        return torch.mean(self.l * rank_loss), \
            torch.mean(self.r * torch.sqrt(mse_loss)), \
            torch.mean(self.l * rank_loss + self.r * torch.sqrt(mse_loss))
