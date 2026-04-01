import sys
import numpy as np

import sys

sys.setrecursionlimit(100000)

max_val = sys.maxsize


def dtw_dis(C, Q):
    """计算两条轨迹/数值序列的 DTW 距离。

    参数:
        C: [m, dim]
        Q: [n, dim]

    返回:
        标量 DTW 距离
    """
    C = np.array(C)
    Q = np.array(Q)
    assert np.size(C, 1) == np.size(Q, 1)
    m = np.size(C, 0)
    n = np.size(Q, 0)
    dim = np.size(C, 1)

    # 点对点欧氏距离矩阵: [m, n]
    point_dis = np.zeros((m, n), dtype="float64")
    for i in range(m):
        for j in range(n):
            point_dis[i, j] = np.sqrt(sum([(C[i, k] - Q[j, k]) ** 2 for k in range(dim)]))

    # 动态规划累计代价矩阵: [m, n]
    warping_dis = np.ones((m, n), dtype="float64") * max_val
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                warping_dis[0, 0] = point_dis[0, 0]
                continue

            if i > 0 and j > 0:
                left_down = warping_dis[i - 1, j - 1]
            else:
                left_down = max_val

            if i > 0:
                down = warping_dis[i - 1, j]
            else:
                down = max_val
            if j > 0:
                left = warping_dis[i, j - 1]
            else:
                left = max_val

            warping_dis[i, j] = point_dis[i, j] + min(left_down, down, left)

    dis = warping_dis[m - 1, n - 1]

    return dis
