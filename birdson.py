#!/usr/bin/env python3 

import numpy as np 
import pandas as pd 


# import numpy as np
from scipy.spatial import cKDTree as KDTree

# data = np.random.rand(90000, 4)


# 计算维度空间中的网格大小
def grid_size(dimensions, min_dist):
    # 计算每个维度需要多少个网格单元
    return tuple(int(np.ceil(d / min_dist)) for d in dimensions)

# 创建一个优化的蓝噪声采样函数，使用网格来加速近邻检查
def optimized_blue_noise_sampling(data, min_dist, ndim):
    # 获取数据的范围
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    dimensions = data_max - data_min
    
    # 初始化一个足够大的网格
    grid = grid_size(dimensions, min_dist)
    occupied_cells = set()

    # 映射点到网格单元
    def get_cell_coords(point):
        return tuple(int((point[i] - data_min[i]) / min_dist) for i in range(ndim))

    # 采样后的数据集
    sampled_data = []

    # 遍历数据集中的每一个点
    for point in data:
        cell_coords = get_cell_coords(point)
        if cell_coords not in occupied_cells:
            # 检查相邻的单元格
            is_empty = True
            for offset in np.ndindex((3,) * ndim):  # 检查3^ndim个相邻单元
                neighbor = tuple(cell_coords[i] + offset[i] - 1 for i in range(ndim))
                if neighbor in occupied_cells:
                    is_empty = False
                    break
            if is_empty:
                sampled_data.append(point)
                occupied_cells.add(cell_coords)

    sampled_data = np.array(sampled_data)
    sampled_data[:, 0] = (sampled_data[:, 0] - 0.5) * 2000.  
    sampled_data[:, 1] = sampled_data[:, 1] * 1000.  
    sampled_data[:, 2] = (sampled_data[:, 2] - 0.5) * 2000.  
    sampled_data[:, 3] = sampled_data[:, 3] * 70.  
    sampled_data[:, 4] = sampled_data[:, 4].astype(int)
    # print(sampled_data)
    return sampled_data

# 使用示例
if __name__ == "__main__":
    # 假设有一个 4x90000 的 numpy 数组
    data = np.random.rand(90000, 4)  # 用随机数代替实际数据
    pdata = pd.read_csv("param.csv")
    pdata = pdata.sort_values(by=["Loglike"], ascending=False)
    data = pd.DataFrame({
        "M1": (pdata['M1'] / 2000.) + 0.5, 
        "M2": (pdata['M2'] / 1000.), 
        "Mu": (pdata['mu'] / 2000.) + 0.5, 
        "TB": (pdata['TanBeta'] / 70.),
        "ID": pdata['index']
    })
    print(pdata)
    data = data.to_numpy()
    ndim = 4  # 数据的维度
    min_dist = 0.02  # 我们想要的大约1000个点的最小距离

    # 执行优化的蓝噪声采样
    sampled_data = optimized_blue_noise_sampling(data, min_dist, ndim)
    # print(data)
    print(sampled_data)
    df = pd.DataFrame(sampled_data, columns=['M1', "M2", "Mu", "Tb", "ID"])
    df.to_csv("param_filtered.csv", index=False)


    print(f"Number of points after sampling: {sampled_data.shape[0]}")
    print(f"Sampled data shape: {sampled_data.shape}")
