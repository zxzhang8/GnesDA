import torch
import random
import numpy as np


def l2_dist(q: np.ndarray, x: np.ndarray):
    assert len(q.shape) == 2
    assert len(x.shape) == 2
    assert q.shape[1] == q.shape[1]
    x = x.T
    sqr_q = np.sum(q ** 2, axis=1, keepdims=True)
    sqr_x = np.sum(x ** 2, axis=0, keepdims=True)
    l2 = sqr_q + sqr_x - 2 * q @ x
    l2[np.nonzero(l2 < 0)] = 0.0
    return np.sqrt(l2)


def l2_dist_separate(q: np.ndarray, x: np.ndarray, embed_channel):
    total_dis = np.zeros((len(q), len(x)))
    length = embed_channel
    for i in range(length):
        tem_q = q[:, i, :]
        tem_x = x[:, i, :]
        total_dis += l2_dist(tem_q, tem_x)
    return total_dis


def arg_sort(q, x, embed_channel):
    if len(q.shape) == 2:
        dists = l2_dist(q, x)
    else:
        dists = l2_dist_separate(q, x, embed_channel)
    return np.argsort(dists)


def intersect_sizes(gs, ids):
    return np.array([len(np.intersect1d(g, list(id))) for g, id in zip(gs, ids)])


def test_recall(X, Q, knn, G, embed_channel):
    ks = [1, 5, 10, 50, 100]
    Ts = [1, 5, 10, 50, 100]
    top_count_list = np.zeros((len(Ts), len(ks)))

    sort_idx = arg_sort(Q, X, embed_channel)

    for i, t in enumerate(Ts):
        ids = sort_idx[:, :t]
        tps = [intersect_sizes(knn[:, :top_k], ids) / float(top_k) for top_k in ks]

        rcs = [np.mean(t) for t in tps]
        top_count_list[i] = rcs

    top_1_counter = 0
    for query_id in range(Q.shape[0]):
        if knn[query_id][0] != sort_idx[query_id][0]:
            top11_true_list = [knn[query_id][0]]
            for id in knn[query_id]:
                if G[query_id][id] == G[query_id][top11_true_list[0]]:
                    top11_true_list.append(id)
            if sort_idx[query_id][0] in top11_true_list:
                top_1_counter += 1
        else:
            top_1_counter += 1
    top_count_list[0][0] = top_1_counter / Q.shape[0]

    print("# Probed \t Items \t", end="")
    for top_k in ks:
        print("top-%d\t" % (top_k), end="")
    print()
    for i, t in enumerate(Ts):
        ids = sort_idx[:, :t]
        items = np.mean([len(id) for id in ids])
        print("%6d \t %6d \t" % (t, items), end="")
        rcs = top_count_list[i]

        # vrs = [np.std(t) for t in tps]
        for rc in rcs:
            print("%.4f \t" % rc, end="")
        # for vr in vrs:
        #     print("%.4f \t" % vr, end="")
        print()
    print()
    # for top_k in ks:
    #     print("top-%d\t" % top_k, end="")
    # print()
    # for top_k in ks:
    #     ids = sort_idx[:, :top_k]
    #     tps = [intersect_sizes(knn[:, :top_k], ids) / float(top_k)]
    #     rcs = [np.mean(t) for t in tps]
    #     for rc in rcs:
    #         print("%.4f \t" % rc, end="")
    # print()


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
