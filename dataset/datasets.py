import torch
import warnings
import numpy as np
from random import randint
from torch.utils.data import Dataset


def word2sig(lines, max_length=None):
    lens = [len(line) for line in lines]
    if max_length is None:
        max_length = np.max(lens)
    elif max_length < np.max(lens):
        warnings.warn("K is {} while strings may " "exceed the maximum length {}".format(max_length, np.max(lens)))

    # all_chars = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
    #              'N': 12, 'O': 13, 'P': 14, 'Q': 15, 'R': 16, 'S': 17, 'T': 18, 'U': 19, 'V': 20, 'W': 21, 'x': 22,
    #              'Y': 23, 'Z': 24}
    all_chars = dict()
    all_chars["counter"] = 0
    alphabet = ''

    def to_ord(c):
        nonlocal all_chars
        nonlocal alphabet
        if not (c in all_chars):
            alphabet += c
            all_chars[c] = all_chars["counter"]
            all_chars["counter"] = all_chars["counter"] + 1
        return all_chars[c]

    x = [[to_ord(c) for c in line] for line in lines]

    return all_chars["counter"], max_length, x, alphabet


class StringDataset(Dataset):

    def __init__(self, C, M, sig, data_type):
        self.C, self.M = C, M
        self.sig = sig
        self.data_type = data_type

    def __getitem__(self, index):
        if self.data_type == "protein":
            encode = np.zeros((self.C, self.M), dtype=np.float32)
            encode[np.array(self.sig[index]), np.arange(len(self.sig[index]))] = 1.0
            return torch.from_numpy(encode)
        else:
            return torch.tensor(self.sig[index], dtype=torch.float32)

    def __len__(self):
        return len(self.sig)


class TripletString(Dataset):
    def __init__(self, strings, lens, knn, dist, K):
        self.lens, self.knn, self.dist = lens, knn, dist
        self.N, self.C, self.M = len(strings), strings.C, strings.M
        self.N, self.K = self.knn.shape
        self.K = min(K, self.K)
        self.strings = strings
        self.index = np.arange(self.N)
        self.avg_dist = np.mean(self.dist)

    def __getitem__(self, idx):
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
            self.strings[anchor],
            self.strings[positive],
            self.strings[negative],
            pos_dist / self.avg_dist,
            neg_dist / self.avg_dist,
            pos_neg_dist / self.avg_dist,
        )

    def __len__(self):
        return self.N

    def update_k(self, new_k):
        self.K = new_k
