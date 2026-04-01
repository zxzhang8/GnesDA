import time
import torch
from tqdm import tqdm
from torch.utils import data
from torch.optim import lr_scheduler

from model.GnesDA import GnesDAModel
from train.network import TripletNet, TripletLoss


def train_epoch(args, train_set, device):
    """训练一个 GnesDA 三元组网络并返回完整模型。"""
    C, M = train_set.C, train_set.M

    torch.manual_seed(time.time())

    # DataLoader 输出:
    #   anchor / pos / neg:
    #       protein: [B, C, M]
    #       traj:    [B, M, 2]
    #   三个真实距离:
    #       [B]
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    train_steps  = len(train_loader)

    if args.embed == "transformer":
        net = GnesDAModel(args, channels=C, max_seq_len=M, embed_len=args.embed_len).to(device)
    else:
        raise ValueError("wrong embed type!!!")

    model  = TripletNet(net, device=device).to(device)
    losser = TripletLoss(args).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                        steps_per_epoch=train_steps,
                                        pct_start=0.3,
                                        epochs=args.epochs,
                                        max_lr=args.learning_rate)

    with tqdm(total=args.epochs * len(train_loader), desc="# training") as p_bar:
        for epoch in range(args.epochs):
            agg   = 0.0
            agg_r = 0
            agg_m = 0
            optimizer.zero_grad()

            start_time = time.time()
            for idx, batch in enumerate(train_loader):
                (
                    anchor, pos, neg,
                    pos_dist, neg_dist, pos_neg_dist,
                ) = (i.to(device) for i in batch)

                optimizer.zero_grad()
                # output 是长度为 3 的 tuple，
                # 每一项来自 GnesDAModel.forward:
                #   一般形状为 [B, C, T]，其中 T = embed_len // C
                output = model((anchor, pos, neg))

                r, m, loss = losser(
                    output,
                    (pos_dist, neg_dist, pos_neg_dist),
                    epoch,
                )

                loss.backward()
                optimizer.step()
                # scheduler.step()

                agg   += loss.item()
                agg_r += r.item()
                agg_m += m.item()
                p_bar.update(1)
                p_bar.set_description(
                    "# Epoch: %3d Time: %.3f Loss: %.4f  r: %.4f m: %.4f"
                    % (
                        epoch,
                        time.time() - start_time,
                        agg / (idx + 1),
                        agg_r / (idx + 1),
                        agg_m / (idx + 1),
                    )
                )

    return model
