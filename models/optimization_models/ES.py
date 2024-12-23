import numpy as np
from multiprocessing import Pool
from functools import partial

from utils.utils import deleteFilesAndFolders

def es(x, y, obs, sd, q, range_, forwardmodel):
    """
    x: 模型参数 (1, m)
    y: 模型输出 (1, g)
    obs: 观测值 (1, g)
    q: ES 算法迭代次数
    range_: 模型参数采样范围 (m, 2)
    forwardmodel: 正演模型
    """
    x = np.array(x)
    y = np.array(y)
    obs = np.array(obs)
    sd = np.array(sd)

    Nobs = obs.shape[0] # 观测值个数 g
    # m = x.shape[0] # 需反演的参数个数 m
    N1 = y.shape[1] # 采样数量

    Cd = np.eye(Nobs) * (sd ** 2) # 观测误差的协方差矩阵

    X = x # 供存储
    Y = y # 供存储

    for i in np.arange(q):
        print("当前第", i, "次迭代")
    
        mean_x = np.tile(np.mean(x, axis=1).reshape(-1, 1), (1, N1)) # mean of the parameters
        mean_y = np.tile(np.mean(y, axis=1).reshape(-1, 1), (1, N1)) # mean of the outputs

        Cxy = (x - mean_x) @ (y - mean_y).T / (N1 - 1)
        Cyy = (y - mean_y) @ (y - mean_y).T / (N1 - 1)
        kgain = Cxy @ np.linalg.inv(Cyy + Cd) # Kalman gain
        obse = np.tile(obs.reshape(-1, 1), N1) + np.random.normal(0, np.tile(sd.reshape(-1, 1), N1))
        x += kgain @ (obse - y) # update
        
        x = np.clip(x, range_[:, 0:1], range_[:, 1:2]) # Boundary handling

        with Pool() as pool:
            args = [(x[:, j], ) for j in range(N1)]
            results = pool.map(partial(forwardmodel_wrapper, forwardmodel=forwardmodel), args)
            y = np.column_stack(results)
        # for j in range(N1):
            # y[:, j] = forwardmodel(x[:, j])

        X = np.hstack((X, x))
        Y = np.hstack((Y, y))

    deleteFilesAndFolders('simulation_folder')

    return X, Y

# 包装函数, 用于并行化计算
def forwardmodel_wrapper(args, forwardmodel):
    return forwardmodel(*args)