import os
import time
import tqdm
import pickle
import argparse
import numpy as np
from multiprocessing import cpu_count

from utils import grid
from utils.function import setup_seed
from utils.fasta import prepare_dna_dataset
from train.embed import GnesDA_embedding
from dataset.datasets import word2sig, StringDataset
from distance.dist_computation import all_pair_distance


def get_knn(dist):
    """对真实距离矩阵逐行排序，得到每个样本的近邻索引。

    参数:
        dist: [num_query, num_base]，元素为真实序列距离。

    返回:
        knn: [num_query, num_base]，每一行是按距离升序排序后的 base 下标。
    """
    knn = np.empty(dtype=np.int32, shape=(len(dist), len(dist[0])))
    for i in tqdm.tqdm(range(len(dist)), desc="# sorting for KNN indices"):
        knn[i, :] = np.argsort(dist[i, :])
    return knn


def get_dist_knn(dist_type, queries, base=None, data_type=None):
    """计算两组序列的全对距离与对应的近邻排序。"""
    if base is None:
        base = queries

    dist = all_pair_distance(queries, base, cpu_count(), dist_type, data_type=data_type)

    return dist, get_knn(dist)


def ReadData_fromfile(dataset, data_type):
    """按数据集名称读取原始序列列表。

    说明:
        - protein / dna 数据返回字符串序列列表。
        - traj 数据返回轨迹列表，单条轨迹通常为 [[x, y], ...]。
    """
    lines = []
    if data_type in ("protein", "dna"):
        if dataset in ("uniprot", "uniref") or os.path.isdir("data/{}".format(dataset)):
            datafile = ["train_seq_list", "query_seq_list", "base_seq_list"]
            for d in datafile:
                lines.extend(pickle.load(open("data/{}/{}".format(dataset, d), "rb")))
        else:
            raise ValueError("wrong dataset type!!!")
    elif dataset == "geolife":
        lines.extend(pickle.load(open("data/0_geolife/traj_list", "rb")))
    elif dataset == "porto":
        lines.extend(pickle.load(open("data/0_porto_all/traj_list", "rb")))
    else:
        raise ValueError("wrong dataset type!!!")
    return lines


class DataHandler:
    def __init__(self, args, data_f):
        """负责数据划分、真实距离加载、以及模型输入构造。

        与论文第 4 节对应:
            1. 先将不同类型序列统一变换为 [D, L] 形式的矩阵输入。
            2. 再交给下游卷积块 + Transformer 块进行编码。
        """
        self.data_f    = data_f
        self.args      = args
        self.nt        = args.nt
        self.nq        = args.nq
        self.maxl      = args.maxl
        self.dataset   = args.dataset
        self.data_type = args.data_type

        self.lines     = ReadData_fromfile(args.dataset, args.data_type)

        # 若指定 maxl，则直接截断原始序列长度；0 表示不截断。
        if self.maxl != 0:
            self.lines = [l[: self.maxl] for l in self.lines]
        self.ni = len(self.lines)
        self.nb = self.ni - self.nq - self.nt

        # 训练/查询/候选库的划分与真实距离矩阵会缓存到 knn/ 目录下，避免重复计算。
        self.load_ids()
        self.load_dist()

        start_time = time.time()
        if self.data_type in ("protein", "dna"):
            # 离散字符序列:
            #   原始输入: 长度可变的字符序列
            #   统一表示: one-hot 前的字符 id 序列，后续在 StringDataset 中转成 [C, M]
            #   C: 字母表大小 |Z|
            #   M: 数据集中最大序列长度
            allowed_chars = None
            fixed_alphabet = None
            if self.data_type == "dna":
                allowed_chars = "ACGT"
                fixed_alphabet = "ACGT"
            self.C, self.M, self.char_ids, self.alphabet = word2sig(
                self.lines,
                max_length=None,
                allowed_chars=allowed_chars,
                fixed_alphabet=fixed_alphabet,
            )
            self.string_t = [self.char_ids[i] for i in self.train_ids]
            self.string_q = [self.char_ids[i] for i in self.query_ids]
            self.string_b = [self.char_ids[i] for i in self.base_ids]
        elif self.data_type == "traj":
            # 轨迹/数值序列:
            #   原始输入: 长度可变的二维点序列 [[lon, lat], ...]
            #   论文中的统一表示: 先做 min-max 风格的坐标平移/缩放，再经 MLP 映射到 D 维
            #   这里先完成几何预处理与 padding，真正的 MLP 投影在 Pretreatment 中实现。
            self.traj_length_list = [len(traj) for traj in self.lines]
            self.M = max(self.traj_length_list)
            self.C = args.embed_channel
            self.lines = grid.split_traj_into_equal_grid(self.lines)
            self.lines = grid.pad_traj_list(args.dist_type, self.lines, self.M, pad_value=1.0)
            self.string_t = [self.lines[i] for i in self.train_ids]
            self.string_q = [self.lines[i] for i in self.query_ids]
            self.string_b = [self.lines[i] for i in self.base_ids]
        print("# Loading time: {}".format(time.time() - start_time))

        self.xt = StringDataset(self.C, self.M, self.string_t, self.data_type)
        self.xq = StringDataset(self.C, self.M, self.string_q, self.data_type)
        self.xb = StringDataset(self.C, self.M, self.string_b, self.data_type)

        print(
            "# Unique signature     : {}".format(self.C),
            "# Maximum length       : {}".format(self.M),
            "# Sampled Train Items  : {}".format(self.nt),
            "# Sampled Query Items  : {}".format(self.nq),
            "# Number of Base Items : {}".format(self.nb),
            "# Number of Items      : {}".format(self.ni),
            "# train dist : {}".format(self.train_knn.shape),
            "# query dist : {}".format(self.query_knn.shape),
            sep="\n",
        )

    def generate_ids(self):
        """按固定切片方式生成训练集、查询集、候选库索引。"""
        idx = np.arange(self.ni)
        self.train_ids = idx[: self.nt]
        self.query_ids = idx[self.nt: self.nq + self.nt]
        self.base_ids  = idx[self.nq + self.nt:]

    def generate_dist(self):
        """计算训练集内部与查询集到候选库之间的真实距离。"""
        self.train_dist, self.train_knn = get_dist_knn(
            self.args.dist_type,
            [self.lines[i] for i in self.train_ids],
            data_type=self.data_type,
        )
        self.query_dist, self.query_knn = get_dist_knn(
            self.args.dist_type,
            [self.lines[i] for i in self.query_ids],
            [self.lines[i] for i in self.base_ids],
            data_type=self.data_type,
        )

    def load_ids(self):
        """从磁盘读取或生成数据划分索引。"""
        idx_dir = "{}/".format(self.data_f)
        if not os.path.isfile(idx_dir + "train_idx.npy"):
            self.generate_ids()
            np.save(idx_dir + "train_idx.npy", self.train_ids)
            np.save(idx_dir + "query_idx.npy", self.query_ids)
            np.save(idx_dir + "base_idx.npy", self.base_ids)
        else:
            print("# loading indices from file")
            self.train_ids = np.load(idx_dir + "train_idx.npy")
            self.query_ids = np.load(idx_dir + "query_idx.npy")
            self.base_ids  = np.load(idx_dir + "base_idx.npy")

    def load_dist(self):
        """从磁盘读取或生成真实距离矩阵与近邻排序。"""
        idx_dir = "{}/".format(self.data_f)
        if not os.path.isfile(idx_dir + "train_dist.npy"):
            self.generate_dist()
            np.save(idx_dir + "train_dist.npy", self.train_dist)
            np.save(idx_dir + "train_knn.npy", self.train_knn)
            np.save(idx_dir + "query_dist.npy", self.query_dist)
            np.save(idx_dir + "query_knn.npy", self.query_knn)
        else:
            print("# loading dist and knn from file")
            self.train_dist = np.load(idx_dir + "train_dist.npy")
            self.train_knn = np.load(idx_dir + "train_knn.npy")
            self.query_dist = np.load(idx_dir + "query_dist.npy")
            self.query_knn = np.load(idx_dir + "query_knn.npy")

    def set_nb(self, nb):
        """按用户要求裁剪候选库大小，并同步更新查询真值。"""
        if nb < len(self.base_ids):
            self.base_ids = self.base_ids[:nb]
            self.query_dist = self.query_dist[:, :nb]
            self.query_knn = get_knn(self.query_dist)
            self.xb.sig = self.xb.sig[:nb]


def get_args():
    """读取命令行参数，并构造数据处理器。

    默认超参数基本对应论文实现:
        - patch_len=16, stride=8
        - e_layers=6
        - protein / dna 的最终嵌入维度约为 C * 5
        - traj 的最终嵌入维度约为 C * 4
    """
    parser = argparse.ArgumentParser(description="HyperParameters for String Embedding")

    parser.add_argument("--data_type",           type=str, default="protein", choices=["protein", "dna", "traj"], help="the type of data: protein, dna or trajtory")
    parser.add_argument("--dataset",             type=str, default="uniprot", help="dataset")
    parser.add_argument("--embed-dir",           type=str, default="", help="embedding save location")
    parser.add_argument("--embed",               type=str, default="transformer", help="embedding method")
    parser.add_argument("--dist_type",           type=str, default="ed", help="distance type")

    parser.add_argument("--nt",                  type=int, default=1000, help="training samples")
    parser.add_argument("--nq",                  type=int, default=1000, help="query items")
    parser.add_argument("--nb",                  type=int, default=5000000, help="base items")
    parser.add_argument("--k",                   type=int, default=200, help="sampling threshold")
    parser.add_argument("--maxl",                type=int, default=0, help="max length of strings")

    parser.add_argument("--epochs",              type=int, default=300, help="epochs")
    parser.add_argument("--shuffle-seed",        type=int, default=666, help="seed for shuffle")
    parser.add_argument("--batch-size",          type=int, default=32, help="batch size for train")
    parser.add_argument("--test-batch-size",     type=int, default=32, help="batch size for test")
    parser.add_argument("--num-workers",         type=int, default=0, help="dataloader workers for train/test")
    parser.add_argument("--channel",             type=int, default=8, help="channels of cnn")
    parser.add_argument("--embed-len",           type=int, default=128, help="output length")
    parser.add_argument("--embed-channel",       type=int, default=32, help="output channel of trajectory dataset")
    parser.add_argument("--learning_rate",       type=float, default=0.001, help="learning rate")

    parser.add_argument("--save-model",          action="store_true", default=False, help="save model")
    parser.add_argument("--save-split",          action="store_true", default=False, help="save split data folder")
    parser.add_argument("--save-embed",          action="store_true", default=False, help="save embedding")
    parser.add_argument("--recall",              action="store_true", default=True, help="print recall")
    parser.add_argument("--distance-correlation", action="store_true", default=True, help="print embedding/edit-distance correlation")
    parser.add_argument("--no-cuda",             action="store_true", default=False, help="disables GPU training")

    parser.add_argument("--train-fasta",         type=str, default="", help="optional training FASTA for DNA")
    parser.add_argument("--eval-fasta",          type=str, default="", help="optional eval FASTA for DNA; split into query/base")
    parser.add_argument("--query-fasta",         type=str, default="", help="optional query FASTA for DNA")
    parser.add_argument("--base-fasta",          type=str, default="", help="optional base FASTA for DNA")
    parser.add_argument("--eval-query-ratio",    type=float, default=0.5, help="fraction of eval FASTA assigned to query")

    # GnesDA
    parser.add_argument('--conv_channels',       type=int, default=10, help='num of conv channels in pretreatment')
    parser.add_argument('--conv_layers',         type=int, default=3, help='num of conv layers in pretreatment')
    parser.add_argument('--e_layers',            type=int, default=6, help='num of encoder layers')
    parser.add_argument('--d_model',             type=int, default=64, help='dimension of transformer model')
    parser.add_argument('--n_heads',             type=int, default=1, help='num of heads')
    parser.add_argument('--d_ff',                type=int, default=256, help='dimension of fcn')
    parser.add_argument('--dropout',             type=float, default=0.05, help='dropout')
    parser.add_argument('--patch_len',           type=int, default=16, help='patch length')
    parser.add_argument('--padding_patch',       type=str, default='end', help='None: None; end: padding on the end')
    parser.add_argument('--stride',              type=int, default=8, help='stride')
    parser.add_argument('--affine',              type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--revin',               type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--subtract_last',       type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--norm',                type=str, default=None, help='transformer norm type')

    args = parser.parse_args()
    if args.data_type == "dna" and args.dist_type not in ("ed", "nw"):
        raise ValueError("DNA only supports 'ed' and 'nw' distance types.")

    if args.data_type == "dna":
        metadata = prepare_dna_dataset(
            dataset=args.dataset,
            train_fasta=args.train_fasta or None,
            eval_fasta=args.eval_fasta or None,
            query_fasta=args.query_fasta or None,
            base_fasta=args.base_fasta or None,
            seed=args.shuffle_seed,
            eval_query_ratio=args.eval_query_ratio,
        )
        if metadata is not None:
            args.nt = metadata["train_size"]
            args.nq = metadata["query_size"]
            args.nb = min(args.nb, metadata["base_size"])
            print(
                "# Prepared DNA dataset from FASTA\n"
                f"# train_size:{metadata['train_size']}\n"
                f"# query_size:{metadata['query_size']}\n"
                f"# base_size:{metadata['base_size']}\n"
                f"# max_sequence_length:{metadata['max_sequence_length']}"
            )

    print(f"d_model:{args.d_model}")
    print(f"e_layers:{args.e_layers}")
    print(f"conv_layers:{args.conv_layers}")
    print(f"n_heads:{args.n_heads}")
    print(f"d_ff:{args.d_ff}")
    print(f"patch_len:{args.patch_len}")
    print(f"epochs:{args.epochs}")
    print(f"embed-len:{args.embed_len}")
    print(f"embed-channel:{args.embed_channel}")

    data_file = "knn/{}/{}/nt{}_nq{}".format(
        args.dataset,
        args.dist_type,
        args.nt,
        args.nq,
    )
    os.makedirs(data_file, exist_ok=True)
    setup_seed(args.shuffle_seed)

    h = DataHandler(args, data_file)
    h.set_nb(args.nb)
    return args, h, data_file


if __name__ == "__main__":
    args, h, data_file = get_args()
    GnesDA_embedding(args, h, data_file)
