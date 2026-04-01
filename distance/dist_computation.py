import time
import Levenshtein
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

from distance.dtw import dtw_dis
from distance.edr import edr_dis
import distance.pairwise_alignment as pa


def f(x, dist_type, data_type):
    """计算单个查询序列到一组候选序列的真实距离。"""
    a, B = x
    if dist_type == "ed":
        return [Levenshtein.distance(a, b) for b in B]
    if dist_type == "nw":
        moltype = "nucl" if data_type == "dna" else "prot"
        tmp = []
        for b in B:
            aln = pa.needle(moltype=moltype, qseq=a, sseq=b, gapextend=0)
            tmp.append(aln.pidentity)
        return tmp
    elif dist_type == "dtw":
        return [dtw_dis(a, b) for b in B]
    elif dist_type == "edr":
        return [edr_dis(a, b) for b in B]
    else:
        raise ValueError("wrong dist type!!!")


def all_pair_distance(A, B, n_thread, dist_type, data_type=None, progress=True):
    """多进程计算两组序列的全对距离矩阵。

    返回:
        [len(A), len(B)]
    """
    bar = tqdm if progress else lambda iterable, total, desc: iterable
    g = partial(f, dist_type=dist_type, data_type=data_type)

    def all_pair(A, B, n_thread):
        with Pool(n_thread) as pool:
            start_time = time.time()
            edit = list(
                bar(
                    pool.imap(g, zip(A, [B for _ in A])),
                    total=len(A),
                    desc="# edit distance {}x{}".format(len(A), len(B)),
                ))
            if progress:
                print("# Calculate edit distance time: {}".format(time.time() - start_time))
            return np.array(edit)

    if len(A) < len(B):
        return all_pair(B, A, n_thread).T
    else:
        return all_pair(A, B, n_thread)
