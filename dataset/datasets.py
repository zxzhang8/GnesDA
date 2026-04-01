import torch
import warnings
import numpy as np
from random import randint
from torch.utils.data import Dataset


def word2sig(lines, max_length=None, allowed_chars=None, fixed_alphabet=None):
    """将字符序列映射为离散 id 序列。

    参数:
        lines: List[str]，原始字符串/蛋白质序列。
        max_length: 允许的最大长度；若为 None，则取数据集最大长度。
        allowed_chars: 允许出现的字符集合；若不为 None，则会严格校验。
        fixed_alphabet: 固定字母表顺序；若给定，则按该顺序分配 id。

    返回:
        C: 字母表大小 |Z|
        M: 最大序列长度
        x: 每条序列的字符 id 列表
        alphabet: 按出现顺序构建的字母表字符串
    """
    lens = [len(line) for line in lines]
    if max_length is None:
        max_length = np.max(lens)
    elif max_length < np.max(lens):
        warnings.warn("K is {} while strings may " "exceed the maximum length {}".format(max_length, np.max(lens)))

    all_chars = dict()
    all_chars["counter"] = 0
    alphabet = ""

    if fixed_alphabet is not None:
        alphabet = fixed_alphabet
        for idx, char in enumerate(fixed_alphabet):
            all_chars[char] = idx
        all_chars["counter"] = len(fixed_alphabet)

    if allowed_chars is not None:
        allowed_chars = set(allowed_chars)

    def validate_line(line_idx, line):
        if allowed_chars is None:
            return
        invalid = sorted(set(line) - allowed_chars)
        if invalid:
            preview = line[:50]
            raise ValueError(
                "Invalid characters {} found in sequence {}: {!r}. Allowed characters are: {}.".format(
                    invalid, line_idx, preview, "".join(sorted(allowed_chars))
                )
            )

    def to_ord(c):
        nonlocal all_chars
        nonlocal alphabet
        if not (c in all_chars):
            alphabet += c
            all_chars[c] = all_chars["counter"]
            all_chars["counter"] = all_chars["counter"] + 1
        return all_chars[c]

    x = []
    for line_idx, line in enumerate(lines):
        validate_line(line_idx, line)
        x.append([to_ord(c) for c in line])

    return all_chars["counter"], max_length, x, alphabet


class StringDataset(Dataset):

    def __init__(self, C, M, sig, data_type):
        """统一的数据集封装。

        对应论文 4.1 节:
            - protein / dna: 输出 one-hot 矩阵 [C, M]
            - traj: 输出 padding 后的连续值矩阵 [M, 2]
              后续会在 Pretreatment 中通过 MLP 投影到 [C, M]
        """
        self.C, self.M = C, M
        self.sig = sig
        self.data_type = data_type

    def __getitem__(self, index):
        if self.data_type in ("protein", "dna"):
            # encode: [C, M]
            #   C: 字母表大小 / 通道数
            #   M: 统一后的最大序列长度
            # 每一列对应序列中的一个位置，每一行对应一个字符类别。
            encode = np.zeros((self.C, self.M), dtype=np.float32)
            encode[np.array(self.sig[index]), np.arange(len(self.sig[index]))] = 1.0
            return torch.from_numpy(encode)
        else:
            # 轨迹输入直接返回 [M, 2]，其中 2 表示 (x, y) / (lon, lat) 两个坐标维。
            return torch.tensor(self.sig[index], dtype=torch.float32)

    def __len__(self):
        return len(self.sig)


class TripletString(Dataset):
    def __init__(self, strings, lens, knn, dist, K):
        """构造三元组训练样本。

        训练监督来自真实距离矩阵:
            - anchor: 当前样本
            - positive / negative: 从近邻集合中随机采样两项
            - 再按真实距离排序，保证 positive 比 negative 更近
        """
        self.lens, self.knn, self.dist = lens, knn, dist
        self.N, self.C, self.M = len(strings), strings.C, strings.M
        self.N, self.K = self.knn.shape
        self.K = min(K, self.K)
        self.strings = strings
        self.index = np.arange(self.N)
        self.avg_dist = np.mean(self.dist)

    def __getitem__(self, idx):
        # anchor / positive / negative 都是单条序列张量:
        #   protein: [C, M]
        #   traj:    [M, 2]
        anchor = idx
        positive = self.knn[anchor, randint(1, min(self.N - 1, self.K))]
        negative = self.knn[anchor, randint(1, min(self.N - 1, self.K))]
        while negative == positive:
            negative = self.knn[anchor, randint(1, min(self.N - 1, self.K))]
        pos_dist = self.dist[anchor, positive]
        neg_dist = self.dist[anchor, negative]
        if pos_dist > neg_dist:
            positive, negative = negative, positive
            pos_dist, neg_dist = neg_dist, pos_dist
        pos_neg_dist = self.dist[positive, negative]

        return (
            # 三个输入序列
            self.strings[anchor],
            self.strings[positive],
            self.strings[negative],
            # 三个标量距离，使用全体均值做归一化，便于稳定训练
            pos_dist / self.avg_dist,
            neg_dist / self.avg_dist,
            pos_neg_dist / self.avg_dist,
        )

    def __len__(self):
        return self.N

    def update_k(self, new_k):
        self.K = new_k
