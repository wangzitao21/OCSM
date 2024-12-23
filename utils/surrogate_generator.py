import numpy as np

from functools import partial
from multiprocessing import Pool

np.set_printoptions(suppress=True)

from models.modflow_model import forwardmodel

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from smt.sampling_methods import LHS, Random, FullFactorial

from utils.utils import deleteFilesAndFolders

# Full-factorial sampling 等份采样
def sampling_factorial(parameter_number, n, _range=np.array([-3, 3])):
    """
    parameter_number: int, 参数数量
    n: int, 抽样数
    """
    xlimits = np.tile(_range, (parameter_number, 1))
    sampling = FullFactorial(xlimits=xlimits)
    x = sampling(n).T

    with Pool() as pool:
        args = [(x[:, k], ) for k in np.arange(n)]
        results = pool.map(partial(forwardmodel_wrapper, forwardmodel=forwardmodel), args)
        y = np.column_stack(results)

    deleteFilesAndFolders('simulation_folder')
    
    return x.T, y.T # 需要转置, 供 smt 使用

# 随机采样
def sampling_random(parameter_number, n, _range=np.array([-3, 3])):
    """
    parameter_number: int, 参数数量
    n: int, 抽样数
    """
    xlimits = np.tile(_range, (parameter_number, 1))
    sampling = Random(xlimits=xlimits)
    x = sampling(n).T

    with Pool() as pool:
        args = [(x[:, k], ) for k in np.arange(n)]
        results = pool.map(partial(forwardmodel_wrapper, forwardmodel=forwardmodel), args)
        y = np.column_stack(results)

    deleteFilesAndFolders('simulation_folder')
    
    return x.T, y.T # 需要转置, 供 smt 使用

# 拉丁超立方采样
def sampling_LHS(parameter_number, n, _range=np.array([-3, 3])):
    """
    parameter_number: int, 参数数量
    n: int, 抽样数
    """
    xlimits = np.tile(_range, (parameter_number, 1))
    sampling = LHS(xlimits=xlimits)
    x = sampling(n).T

    with Pool() as pool:
        args = [(x[:, k], ) for k in np.arange(n)]
        results = pool.map(partial(forwardmodel_wrapper, forwardmodel=forwardmodel), args)
        y = np.column_stack(results)

    deleteFilesAndFolders('simulation_folder')
    
    return x.T, y.T # 需要转置, 供 smt 使用

# def sampling_LHS(parameter_number, n, _range=np.array([-3, 3])):
#     """
#     parameter_number: int, 参数数量
#     n: int, 抽样数
#     """
#     xlimits = np.tile(_range, (parameter_number, 1))
#     sampling = LHS(xlimits=xlimits)
#     x = sampling(n).T

#     args = [(x[:, k], ) for k in np.arange(n)]
#     results = [partial(forwardmodel_wrapper, forwardmodel=forwardmodel)(arg) for arg in args]
#     y = np.column_stack(results)

#     deleteFilesAndFolders('simulation_folder')
    
#     return x.T, y.T


# 高斯采样
def sampling_normal(parameter_number, n):
    # x = np.random.randn(parameter_number, n)
    x = np.random.normal(loc=0.0, scale=1.0, size=(parameter_number, n))

    # 并行
    with Pool() as pool:
        args = [(x[:, k], ) for k in np.arange(n)]
        results = pool.map(partial(forwardmodel_wrapper, forwardmodel=forwardmodel), args)
        y = np.column_stack(results)

    deleteFilesAndFolders('simulation_folder')

    return x.T, y.T # 需要转置, 供 smt 使用

# 均匀采样
def sampling_uniform(parameter_number, n, _range=np.array([-2.5, 2.5])):
    """
    parameter_number: int, 参数数量
    n: int, 抽样数
    """
    x = np.random.uniform(low=_range[0], high=_range[1], size=(parameter_number, n))

    # 并行
    with Pool() as pool:
        args = [(x[:, k], ) for k in np.arange(n)]
        results = pool.map(partial(forwardmodel_wrapper, forwardmodel=forwardmodel), args)
        y = np.column_stack(results)

    deleteFilesAndFolders('simulation_folder')

    return x.T, y.T # 需要转置, 供 smt 使用

# 包装函数, 用于并行化计算
def forwardmodel_wrapper(args, forwardmodel):
    return forwardmodel(*args)