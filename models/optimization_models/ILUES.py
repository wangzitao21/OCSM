import numpy as np
from functools import partial
from multiprocessing import Pool

from utils.utils import deleteFilesAndFolders

def ilues(x, y, obs, sd, q, range_, forwardmodel, *, alpha=0.1):
    """
    参考了张江江的文献, 并基于其 matlab 代码修改

    x: 模型参数 (1, m)
    y: 模型输出 (1, g)
    obs: 观测值 (1, g)
    q: 算法迭代次数
    range_: 模型参数采样范围 (m, 2)
    forwardmodel: 正演模型
    alpha: 膨胀系数
    """
    x = np.array(x)
    y = np.array(y)
    obs = np.array(obs)
    sd = np.array(sd)

    N_Iter = q
    xf = x
    yf = y
    
    Nobs = obs.shape[0] # 观测值个数
    Npar = xf.shape[0] # 参数的数量
    Ne = xf.shape[1] # ensemble size 采了多少样品
    
    Cd = np.eye(Nobs) * (sd ** 2)  # 单位矩阵，观测误差的协方差矩阵
    invCd = np.linalg.inv(Cd) # 避免重复计算

    factor = np.ones((N_Iter, 1)) * np.sqrt(N_Iter) # 

    xall = xf # 存储每个迭代的结果
    yall = yf 
    xa = np.full_like(xf, np.nan) # 用于存储更新的参数
    ya = np.full_like(yf, np.nan) # 用于存储更新的结果

    meanxf = np.tile(np.mean(xf, axis=1).reshape(-1,1), (1, Ne)) # mean of the prior parameters
    Cm = (xf - meanxf) @ ((xf - meanxf).T) / (Ne - 1)  # auto-covariance of the prior parameters

    for n_i in np.arange(N_Iter):
        print("当前第", n_i, "次迭代")

        # 计算 J1
        # J1 = np.full((Ne, 1), np.nan)
        # for i in np.arange(Ne):
            # J1[i, 0] = (yf[:,i] - obs).T @ (invCd) @ (yf[:,i] - obs)
        # 上面三行代码改成以下形式
        # J1 = np.sum(((yf - obs.reshape((-1, 1))) ** 2) * invCd, axis=0).reshape((-1, 1))
        Err = yf - obs.reshape((-1, 1))
        J1 = np.sum(Err * (invCd @ Err), axis=0).reshape((-1, 1))

        beta = factor[n_i]
        # for j in np.arange(Ne):
            # xa[:,j] = local_update(xf, yf, Cm, sd, range_, obs, alpha, beta, J1, j)
        with Pool() as pool:
            args1 = [(xf, yf, Cm, sd, range_, obs, alpha, beta, J1, j) for j in np.arange(Ne)]
            results = pool.map(local_update_wrapper, args1)
            xa = np.column_stack(results)
        # print(xa)
        print("计算正演模型中 ...")
        # for k in np.arange(Ne):
        #     ya[:,k] = forwardmodel(xa[:, k])
        with Pool() as pool:
            args = [(xa[:, k], ) for k in np.arange(Ne)]
            results = pool.map(partial(forwardmodel_wrapper, forwardmodel=forwardmodel), args)
            ya = np.column_stack(results)

        if Npar > 10:
            likf = Cal_Log_Lik(yf, obs, sd)
            lika = Cal_Log_Lik(ya, obs, sd)
            cc = (np.exp(lika - likf)) < np.random.rand(Ne)
            xa[:,cc] = xf[:,cc]
            ya[:,cc] = yf[:,cc]

        xall = np.hstack((xall, xa))
        yall = np.hstack((yall, ya))
        xf = xa
        yf = ya

        deleteFilesAndFolders('simulation_folder')
        
        print("第", n_i, "次迭代已结束")
    
    return xall, yall

# 包装函数, 用于并行化计算
def forwardmodel_wrapper(args, forwardmodel):
    return forwardmodel(*args)

# 包装函数, 用于并行化计算
def local_update_wrapper(args):
    return local_update(*args)

def local_update(xf, yf, Cm, sd, range_, obs, alpha, beta, J1, jj):
    Ne = xf.shape[1] # 采的样品数量

    # J2 = np.full((Ne, 1), np.nan)
    # for i in np.arange(Ne):
        # J2[i,0] = (xf[:,i] - xf[:,jj]).T @ np.linalg.inv(Cm) @ (xf[:,i] - xf[:,jj])
    # 上面三行代码改成以下形式
    J2 = np.sum((xf - xf[:, jj].reshape((-1, 1))) * (np.linalg.inv(Cm) @ (xf - xf[:, jj].reshape((-1, 1)))), axis=0).reshape((-1, 1))

    J3 = J1 / np.max(J1) + J2 / np.max(J2)
    a3 = np.argsort(J3, axis=0) # a3 是排序后的矩阵
    a3 = np.squeeze(a3) # a3 要去掉一层不然有问题

    M = int(np.ceil(Ne * alpha)) # 向上取整
    xl = xf[:, a3[0:M]]
    yl = yf[:, a3[0:M]]

    xu = updatapara(xl, yl, range_, sd * beta, obs)
    xest = xu[:,np.random.permutation(M)[0]]

    return xest

def updatapara(xf, yf, range_, sd, obs):
    obs = np.array(obs) # 观测值的个数
    Nobs = obs.shape[0]
    # Npar = xf.shape[0] # 参数个数 * alpha
    Ne = xf.shape[1] # 参数采样的数量 * alpha

    Cd = np.eye(Nobs) * (sd ** 2)  # 单位矩阵，观测误差的协方差矩阵

    meanxf = np.tile(np.mean(xf, axis=1).reshape(-1,1), (1, Ne))
    meanyf = np.tile(np.mean(yf, axis=1).reshape(-1,1), (1, Ne))

    Cxy = (xf - meanxf) @ (yf - meanyf).T / (Ne - 1)
    Cyy = (yf - meanyf) @ (yf - meanyf).T / (Ne - 1)
    kgain = Cxy @ np.linalg.inv(Cyy + Cd)

    obse = np.tile(obs.reshape((-1, 1)), (1, Ne)) + np.random.normal(loc=0.0, scale=sd, size=(Nobs, Ne))
    xa = xf + kgain @ (obse - yf)

    # Boundary handling
    # for i in range(Ne):
    #     for j in range(Npar):
    #         if xa[j, i] > range_[j, 1]:
    #             xa[j, i] = (range_[j, 1] + xf[j, i]) / 2
    #         elif xa[j, i] < range_[j, 0]:
    #             xa[j, i] = (range_[j, 0] + xf[j, i]) / 2
    # 上述几行改成以下形式
    xa = np.clip(xa, range_[:, 0:1], range_[:, 1:2])
    return xa

def Cal_Log_Lik(y1, obs, sd):
    N = y1.shape[1]
    Lik = np.zeros((N,))
    sd = np.array(sd)
    
    # for i in range(N):
    #     Err = obs - y1[:,i]
    #     Lik[i] = -(len(Err) / 2) * np.log(2 * np.pi) - np.sum(np.log(sd)) - \
    #         0.5 * np.sum((Err / sd)**2)
    
    # Lik = -(Err.shape[0] / 2) * np.log(2 * np.pi) - np.sum(np.log(sd)) - \
    #       0.5 * np.sum((Err / sd) ** 2, axis=0)
    Err = obs.reshape((-1, 1)) - y1
    Lik = -(Err.shape[0] / 2) * np.log(2 * np.pi) - np.sum(np.log(sd)) - \
      0.5 * np.sum((Err / sd.reshape((-1, 1))) ** 2, axis=0)

    return Lik