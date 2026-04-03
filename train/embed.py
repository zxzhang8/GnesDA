import os
import json
import time
import tqdm
import torch
import numpy as np
from torch.utils import data

from utils.function import test_recall, test_distance_correlation
from train.trainer import train_epoch
from dataset.datasets import TripletString, StringDataset


def _batch_embed(args, net, vecs: StringDataset, device):
    """分批将整个数据集编码为 embedding。

    返回:
        protein / traj 均为 [N, C, T]
        其中 C 是统一后的通道数，T = embed_len // C
    """
    test_loader = torch.utils.data.DataLoader(
        vecs,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    net.eval()
    embedding = []
    if args.quiet:
        for x in test_loader:
            embedding.append(net(x.to(device)).cpu().data.numpy())
    else:
        with tqdm.tqdm(total=len(test_loader), desc="# batch embedding") as p_bar:
            for i, x in enumerate(test_loader):
                p_bar.update(1)
                # net(x) 输出:
                #   [B, C, T]
                embedding.append(net(x.to(device)).cpu().data.numpy())
    return np.concatenate(embedding, axis=0)


def GnesDA_embedding(args, h, data_file):
    """训练 GnesDA，并对 train / base / query 三部分数据做编码与检索评估。"""
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_loader = TripletString(h.xt, h.nt, h.train_knn, h.train_dist, K=args.k)

    def periodic_eval(model, epoch):
        model.eval()
        with torch.no_grad():
            xb = _batch_embed(args, model.embedding_net, h.xb, device)
            xq = _batch_embed(args, model.embedding_net, h.xq, device)
        if args.recall:
            test_recall(xb, xq, h.query_knn, h.query_dist, h.C)
        if args.distance_correlation:
            test_distance_correlation(xb, xq, h.query_dist, h.C, avg_dist=train_loader.avg_dist)
        model.train()

    model_file = "{}/model.torch".format(data_file)
    if os.path.isfile(model_file):
        model = torch.load(model_file, weights_only=False)
    else:
        start_time = time.time()
        model = train_epoch(args, train_loader, device, eval_fn=periodic_eval)
        if args.save_model:
            torch.save(model, model_file)
        train_time = time.time() - start_time
        print("# Training time: " + str(train_time))

    model.eval()
    with torch.no_grad():
        xt = _batch_embed(args, model.embedding_net, h.xt, device)
        start_time = time.time()
        xb = _batch_embed(args, model.embedding_net, h.xb, device)
        embed_time = time.time() - start_time
        xq = _batch_embed(args, model.embedding_net, h.xq, device)
        print("# Embedding time: " + str(embed_time))
    if args.save_embed:
        if args.embed_dir != "":
            args.embed_dir = args.embed_dir + "/"
        os.makedirs("{}/{}".format(data_file, args.embed_dir), exist_ok=True)
        np.save("{}/{}embedding_xb".format(data_file, args.embed_dir), xb)
        np.save("{}/{}embedding_xt".format(data_file, args.embed_dir), xt)
        np.save("{}/{}embedding_xq".format(data_file, args.embed_dir), xq)

    if args.recall:
        test_recall(xb, xq, h.query_knn, h.query_dist, h.C)
    if args.distance_correlation:
        metrics = test_distance_correlation(xb, xq, h.query_dist, h.C, avg_dist=train_loader.avg_dist)
        if args.save_embed:
            np.save("{}/{}embedding_query_base_distance".format(data_file, args.embed_dir), metrics["pred_dist"])
            with open("{}/{}distance_metrics.json".format(data_file, args.embed_dir), "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "pearson": metrics["pearson"],
                        "spearman": metrics["spearman"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "avg_diff": metrics["avg_diff"],
                        "max_diff": metrics["max_diff"],
                    },
                    handle,
                    indent=2,
                )
