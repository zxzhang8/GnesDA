import os
import time
import tqdm
import pickle
import json
import math
import argparse
import shutil
import numpy as np
from multiprocessing import cpu_count

from utils import grid
from utils.function import setup_seed
from utils.fasta import prepare_dna_dataset
from train.embed import GnesDA_embedding
from dataset.datasets import word2sig, StringDataset
from distance.dist_computation import all_pair_distance
from utils.sequence_store import CombinedSequenceStore, open_split_store


def get_knn(dist, quiet=False):
    """对真实距离矩阵逐行排序，得到每个样本的近邻索引。

    参数:
        dist: [num_query, num_base]，元素为真实序列距离。

    返回:
        knn: [num_query, num_base]，每一行是按距离升序排序后的 base 下标。
    """
    knn = np.empty(dtype=np.int32, shape=(len(dist), len(dist[0])))
    iterator = range(len(dist))
    if not quiet:
        iterator = tqdm.tqdm(iterator, desc="# sorting for KNN indices")
    for i in iterator:
        knn[i, :] = np.argsort(dist[i, :])
    return knn


def get_dist_knn(dist_type, queries, base=None, data_type=None, quiet=False):
    """计算两组序列的全对距离与对应的近邻排序。"""
    if base is None:
        base = queries

    dist = all_pair_distance(queries, base, cpu_count(), dist_type, data_type=data_type, progress=not quiet)

    return dist, get_knn(dist, quiet=quiet)


def load_dataset_metadata(dataset):
    metadata_path = os.path.join("data", dataset, "metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_dataset_item_count(dataset, data_type):
    metadata = load_dataset_metadata(dataset)
    if metadata is not None and metadata.get("storage_format") == "seqbin_v1":
        return metadata["train_size"] + metadata["query_size"] + metadata["base_size"]
    return len(ReadData_fromfile(dataset, data_type))


def load_sequence_store(dataset, max_length=0):
    dataset_dir = os.path.join("data", dataset)
    stores = {
        "train": open_split_store(dataset_dir, "train"),
        "query": open_split_store(dataset_dir, "query"),
        "base": open_split_store(dataset_dir, "base"),
    }
    return CombinedSequenceStore(stores, split_order=["train", "query", "base"], max_length=max_length)


def iter_store_chunks(store, indices, chunk_size):
    for start in range(0, len(indices), chunk_size):
        chunk_indices = indices[start:start + chunk_size]
        yield start, list(store.iter_indices(chunk_indices))


def ReadData_fromfile(dataset, data_type):
    """按数据集名称读取原始序列列表。

    说明:
        - protein / dna 数据返回字符串序列列表。
        - traj 数据返回轨迹列表，单条轨迹通常为 [[x, y], ...]。
    """
    lines = []
    if data_type in ("protein", "dna"):
        if dataset in ("uniprot", "uniref") or os.path.isdir("data/{}".format(dataset)):
            metadata = load_dataset_metadata(dataset)
            if data_type == "dna" and metadata is not None and metadata.get("storage_format") == "seqbin_v1":
                store = load_sequence_store(dataset)
                try:
                    lines.extend(store.iter_indices(range(len(store))))
                finally:
                    store.close()
            else:
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


def format_bytes(num_bytes):
    """将字节数格式化为易读字符串。"""
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def estimate_knn_cache_size(train_count, query_count, base_count, dist_itemsize=8, knn_itemsize=4):
    """估算首次生成距离矩阵与 KNN 缓存时的磁盘占用。"""
    train_entries = train_count * train_count
    query_entries = query_count * base_count

    train_dist_bytes = train_entries * dist_itemsize
    train_knn_bytes = train_entries * knn_itemsize
    query_dist_bytes = query_entries * dist_itemsize
    query_knn_bytes = query_entries * knn_itemsize

    return {
        "train_dist_bytes": train_dist_bytes,
        "train_knn_bytes": train_knn_bytes,
        "query_dist_bytes": query_dist_bytes,
        "query_knn_bytes": query_knn_bytes,
        "total_bytes": train_dist_bytes + train_knn_bytes + query_dist_bytes + query_knn_bytes,
    }


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
        self.nb        = args.nb
        self.maxl      = args.maxl
        self.dataset   = args.dataset
        self.data_type = args.data_type
        self.metadata  = load_dataset_metadata(args.dataset)
        self.sequence_store = None

        if self.data_type == "dna" and self.metadata is not None and self.metadata.get("storage_format") == "seqbin_v1":
            self.sequence_store = load_sequence_store(args.dataset, max_length=self.maxl)
            self.ni = len(self.sequence_store)
        else:
            self.lines = ReadData_fromfile(args.dataset, args.data_type)
            # 若指定 maxl，则直接截断原始序列长度；0 表示不截断。
            if self.maxl != 0:
                self.lines = [l[: self.maxl] for l in self.lines]
            self.ni = len(self.lines)
        if self.nt + self.nq + self.nb > self.ni:
            raise ValueError(
                "Dataset '{}' contains {} sequences, but nt + nq + nb = {} exceeds it.".format(
                    self.dataset, self.ni, self.nt + self.nq + self.nb
                )
            )

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
            if self.data_type == "dna" and self.sequence_store is not None:
                self.C = 4
                self.alphabet = "ACGT"
                if self.metadata is None:
                    raise ValueError("DNA seqbin dataset requires metadata.json")
                self.M = self.metadata["max_sequence_length"]
                if self.maxl != 0:
                    self.M = min(self.M, self.maxl)
                self.xt = StringDataset(
                    self.C,
                    self.M,
                    None,
                    self.data_type,
                    sequence_store=self.sequence_store,
                    sample_indices=self.train_ids,
                    fixed_alphabet="ACGT",
                )
                self.xq = StringDataset(
                    self.C,
                    self.M,
                    None,
                    self.data_type,
                    sequence_store=self.sequence_store,
                    sample_indices=self.query_ids,
                    fixed_alphabet="ACGT",
                )
                self.xb = StringDataset(
                    self.C,
                    self.M,
                    None,
                    self.data_type,
                    sequence_store=self.sequence_store,
                    sample_indices=self.base_ids,
                    fixed_alphabet="ACGT",
                )
            else:
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
                self.xt = StringDataset(self.C, self.M, self.string_t, self.data_type)
                self.xq = StringDataset(self.C, self.M, self.string_q, self.data_type)
                self.xb = StringDataset(self.C, self.M, self.string_b, self.data_type)
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
            self.xt = StringDataset(self.C, self.M, self.string_t, self.data_type)
            self.xq = StringDataset(self.C, self.M, self.string_q, self.data_type)
            self.xb = StringDataset(self.C, self.M, self.string_b, self.data_type)
        print("# Loading time: {}".format(time.time() - start_time))

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
        self.base_ids  = idx[self.nq + self.nt: self.nq + self.nt + self.nb]

    def validate_ids(self):
        """校验缓存索引是否与当前 nt / nq / nb 配置一致。"""
        expected_lengths = {
            "train_ids": self.nt,
            "query_ids": self.nq,
            "base_ids": self.nb,
        }
        actual_lengths = {
            "train_ids": len(self.train_ids),
            "query_ids": len(self.query_ids),
            "base_ids": len(self.base_ids),
        }
        for name, expected in expected_lengths.items():
            actual = actual_lengths[name]
            if actual != expected:
                raise ValueError(
                    "Cached {} length {} does not match expected {}.".format(name, actual, expected)
                )

    def validate_dist(self):
        """校验缓存矩阵 shape 是否与当前配置一致。"""
        expected_shapes = {
            "train_dist": (self.nt, self.nt),
            "train_knn": (self.nt, self.nt),
            "query_dist": (self.nq, self.nb),
            "query_knn": (self.nq, self.nb),
        }
        actual_shapes = {
            "train_dist": self.train_dist.shape,
            "train_knn": self.train_knn.shape,
            "query_dist": self.query_dist.shape,
            "query_knn": self.query_knn.shape,
        }
        for name, expected in expected_shapes.items():
            actual = actual_shapes[name]
            if actual != expected:
                raise ValueError(
                    "Cached {} shape {} does not match expected {}.".format(name, actual, expected)
                )

    def generate_dist(self):
        """计算训练集内部与查询集到候选库之间的真实距离。"""
        if self.sequence_store is None:
            self.train_dist, self.train_knn = get_dist_knn(
                self.args.dist_type,
                [self.lines[i] for i in self.train_ids],
                data_type=self.data_type,
                quiet=self.args.quiet,
            )
            self.query_dist, self.query_knn = get_dist_knn(
                self.args.dist_type,
                [self.lines[i] for i in self.query_ids],
                [self.lines[i] for i in self.base_ids],
                data_type=self.data_type,
                quiet=self.args.quiet,
            )
            return

        train_dist_path = os.path.join(self.data_f, "train_dist.npy")
        train_knn_path = os.path.join(self.data_f, "train_knn.npy")
        query_dist_path = os.path.join(self.data_f, "query_dist.npy")
        query_knn_path = os.path.join(self.data_f, "query_knn.npy")
        self.train_dist = np.lib.format.open_memmap(train_dist_path, mode="w+", dtype=np.float64, shape=(self.nt, self.nt))
        self.train_knn = np.lib.format.open_memmap(train_knn_path, mode="w+", dtype=np.int32, shape=(self.nt, self.nt))
        self.query_dist = np.lib.format.open_memmap(query_dist_path, mode="w+", dtype=np.float64, shape=(self.nq, self.nb))
        self.query_knn = np.lib.format.open_memmap(query_knn_path, mode="w+", dtype=np.int32, shape=(self.nq, self.nb))

        train_ids = np.asarray(self.train_ids)
        query_ids = np.asarray(self.query_ids)
        base_ids = np.asarray(self.base_ids)

        train_chunk = max(1, min(128, self.nt))
        base_chunk = max(1, min(2048, self.nb if self.nb else 1))

        self._generate_dist_blockwise(
            target_dist=self.train_dist,
            target_knn=self.train_knn,
            query_indices=train_ids,
            base_indices=train_ids,
            query_chunk_size=train_chunk,
            base_chunk_size=max(1, min(2048, self.nt)),
            desc_prefix="train",
        )
        self._generate_dist_blockwise(
            target_dist=self.query_dist,
            target_knn=self.query_knn,
            query_indices=query_ids,
            base_indices=base_ids,
            query_chunk_size=max(1, min(128, self.nq)),
            base_chunk_size=base_chunk,
            desc_prefix="query",
        )
        self.train_dist.flush()
        self.train_knn.flush()
        self.query_dist.flush()
        self.query_knn.flush()

    def _generate_dist_blockwise(
        self,
        target_dist,
        target_knn,
        query_indices,
        base_indices,
        query_chunk_size,
        base_chunk_size,
        desc_prefix,
    ):
        iterator = iter_store_chunks(self.sequence_store, query_indices, query_chunk_size)
        if not self.args.quiet:
            iterator = tqdm.tqdm(
                iterator,
                total=(len(query_indices) + query_chunk_size - 1) // query_chunk_size,
                desc=f"# {desc_prefix} distance chunks",
            )
        for query_start, query_chunk in iterator:
            query_stop = query_start + len(query_chunk)
            for base_start, base_chunk in iter_store_chunks(self.sequence_store, base_indices, base_chunk_size):
                block = all_pair_distance(
                    query_chunk,
                    base_chunk,
                    cpu_count(),
                    self.args.dist_type,
                    data_type=self.data_type,
                    progress=False,
                )
                base_stop = base_start + len(base_chunk)
                target_dist[query_start:query_stop, base_start:base_stop] = block
            for row_id in range(query_start, query_stop):
                target_knn[row_id, :] = np.argsort(target_dist[row_id, :])

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
            try:
                self.validate_ids()
            except ValueError as exc:
                print("# cached indices invalid: {}".format(exc))
                self.generate_ids()
                np.save(idx_dir + "train_idx.npy", self.train_ids)
                np.save(idx_dir + "query_idx.npy", self.query_ids)
                np.save(idx_dir + "base_idx.npy", self.base_ids)

    def load_dist(self):
        """从磁盘读取或生成真实距离矩阵与近邻排序。"""
        idx_dir = "{}/".format(self.data_f)
        if not os.path.isfile(idx_dir + "train_dist.npy"):
            estimate = estimate_knn_cache_size(
                train_count=len(self.train_ids),
                query_count=len(self.query_ids),
                base_count=len(self.base_ids),
            )
            _, _, free_bytes = shutil.disk_usage(idx_dir)
            print(
                "# Estimated disk usage before distance generation\n"
                f"# train_dist.npy  : {format_bytes(estimate['train_dist_bytes'])}\n"
                f"# train_knn.npy   : {format_bytes(estimate['train_knn_bytes'])}\n"
                f"# query_dist.npy  : {format_bytes(estimate['query_dist_bytes'])}\n"
                f"# query_knn.npy   : {format_bytes(estimate['query_knn_bytes'])}\n"
                f"# total_estimated : {format_bytes(estimate['total_bytes'])}\n"
                f"# disk_free       : {format_bytes(free_bytes)}"
            )
            if estimate["total_bytes"] > free_bytes:
                print(
                    "# Warning: estimated cache size exceeds currently available disk space. "
                    "Distance generation is likely to fail."
                )
            self.generate_dist()
        else:
            print("# loading dist and knn from file")
            self.train_dist = np.load(idx_dir + "train_dist.npy", mmap_mode="r")
            self.train_knn = np.load(idx_dir + "train_knn.npy", mmap_mode="r")
            self.query_dist = np.load(idx_dir + "query_dist.npy", mmap_mode="r")
            self.query_knn = np.load(idx_dir + "query_knn.npy", mmap_mode="r")
            try:
                self.validate_dist()
            except ValueError as exc:
                print("# cached dist/knn invalid: {}".format(exc))
                self.generate_dist()

    def set_nb(self, nb):
        """按用户要求裁剪候选库大小，并同步更新查询真值。"""
        if nb < len(self.base_ids):
            self.base_ids = self.base_ids[:nb]
            self.query_dist = self.query_dist[:, :nb]
            self.query_knn = get_knn(self.query_dist, quiet=self.args.quiet)
            if getattr(self.xb, "sample_indices", None) is not None:
                self.xb.sample_indices = self.xb.sample_indices[:nb]
            else:
                self.xb.sig = self.xb.sig[:nb]
            self.nb = nb


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
    parser.add_argument("--sample-size",         type=int, default=0, help="optional train sample count; query/base use ceil(sample-size * 0.25)")
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
    parser.add_argument("--quiet",               action="store_true", default=False, help="disable tqdm progress bars")

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
    parser.add_argument('--patch_len',           type=int, default=6, help='patch length')
    parser.add_argument('--padding_patch',       type=str, default='end', help='None: None; end: padding on the end')
    parser.add_argument('--stride',              type=int, default=3, help='stride')
    parser.add_argument('--affine',              type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--revin',               type=int, default=0, help='RevIN; True 1 False 0')
    parser.add_argument('--subtract_last',       type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--norm',                type=str, default=None, help='transformer norm type')

    args = parser.parse_args()
    if args.data_type == "dna" and args.dist_type not in ("ed", "nw"):
        raise ValueError("DNA only supports 'ed' and 'nw' distance types.")
    if args.sample_size < 0:
        raise ValueError("--sample-size must be non-negative.")

    metadata = None
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

    if args.sample_size:
        args.nt = args.sample_size
        eval_sample_size = math.ceil(args.sample_size * 0.25)
        args.nq = eval_sample_size
        args.nb = eval_sample_size

        if metadata is not None:
            total_items = metadata["train_size"] + metadata["query_size"] + metadata["base_size"]
        else:
            total_items = get_dataset_item_count(args.dataset, args.data_type)
        required_items = args.nt + args.nq + args.nb
        if total_items < required_items:
            raise ValueError(
                "Dataset '{}' contains {} sequences, but --sample-size {} requires at least {} "
                "(train={}, query={}, base={}).".format(
                    args.dataset, total_items, args.sample_size, required_items, args.nt, args.nq, args.nb
                )
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

    data_file = "knn/{}/{}/nt{}_nq{}_nb{}".format(
        args.dataset,
        args.dist_type,
        args.nt,
        args.nq,
        args.nb,
    )
    os.makedirs(data_file, exist_ok=True)
    setup_seed(args.shuffle_seed)

    h = DataHandler(args, data_file)
    h.set_nb(args.nb)
    return args, h, data_file


if __name__ == "__main__":
    args, h, data_file = get_args()
    GnesDA_embedding(args, h, data_file)
