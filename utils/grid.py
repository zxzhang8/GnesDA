import numpy as np

GRID_SIZE = 0.002


def _equal_grid(one_dim_list):
    """统计单个坐标维度的整体取值范围。"""
    min_value_list = [np.min(value) for value in one_dim_list]
    max_value_list = [np.max(value) for value in one_dim_list]

    min_value = np.min(min_value_list)
    max_value = np.max(max_value_list)

    grid_size = GRID_SIZE

    max_grid_id = int((max_value - min_value) / grid_size)

    return max_grid_id, min_value, max_value


def split_traj_into_equal_grid(traj_list):
    """对轨迹做统一的坐标平移/缩放预处理。

    参数:
        traj_list: List[[[lon, lat], ...]]

    返回:
        处理后的轨迹列表，单个点仍是二维向量 [x, y]。

    说明:
        论文中数值序列先做预处理再送入 MLP。
        这里的实现并非严格 min-max 到 [0,1]，而是减去全局最小值后再放缩。
    """
    lon_list = []
    lat_list = []
    for traj in traj_list:
        tem_lon = []
        tem_lat = []
        for point in traj:
            tem_lon.append(point[0])
            tem_lat.append(point[1])
        lon_list.append(tem_lon)
        lat_list.append(tem_lat)

    lon_max_grid_id, lon_min_value, lon_max_value = _equal_grid(lon_list)
    lat_max_grid_id, lat_min_value, lat_max_value = _equal_grid(lat_list)

    # grid_size = GRID_SIZE

    tem_traj_list = []
    for traj in traj_list:
        tem_traj = []
        for point in traj:
            p0 = (point[0] - lon_min_value) * 10
            p1 = (point[1] - lat_min_value) * 10
            tem_traj.append([p0, p1])
        tem_traj_list.append(tem_traj)
    # for i in range(len(traj_list)):
    #     for j in range(len(traj_list[i])):
    #         traj_list[i][j][0] = (traj_list[i][j][0] - lon_min_value) * 10
    #         traj_list[i][j][1] = (traj_list[i][j][1] - lat_min_value) * 10

    return tem_traj_list


def pad_traj_list(dist_type, seq_list, max_length, pad_value=0.0):
    """将所有轨迹补齐到统一长度。

    输出中每条轨迹形状为 [max_length, 2]。
    对 DTW，padding 值使用最后一个真实点，避免尾部突变过大。
    """
    value = [1.0 * pad_value for i in range(len(seq_list[0][0]))]
    final_pad_seq_list = []
    for seq in seq_list:
        assert len(seq) <= max_length, "Sequence length {} is larger than max_length {}".format(len(seq), max_length)

        if dist_type == "dtw":
            value = [1.0 * val for val in seq[len(seq) - 1]]
        for j in range(max_length - len(seq)):
            seq.append(value)
        final_pad_seq_list.append(seq)
    return final_pad_seq_list
