from sko.GA import GA

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from smt.surrogate_models import KRG, KPLS, KPLSK
from sklearn.metrics import r2_score
np.set_printoptions(suppress=True)
from utils.surrogate_generator import sampling_LHS, sampling_normal, sampling_uniform, sampling_random, sampling_factorial

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import pickle

from joblib import Parallel, delayed

parameter_number = 34
train_numbers = 1024

# ! ######################################################################

print("现在加载测试集 ...")

x_test_normal = np.load("./data/x_test_normal.npy")
y_test_normal = np.load("./data/y_test_normal.npy")

x_test_uniform = np.load("./data/x_test_uniform.npy")
y_test_uniform = np.load("./data/y_test_uniform.npy")

x_test = np.vstack((x_test_normal, x_test_uniform))
y_test = np.vstack((y_test_normal, y_test_uniform))

print("测试集已加载 ...")

# ! ######################################################################

def test_acquired(sm, x_test=x_test, y_test=y_test):
    y_test_predicted = sm.predict_values(x_test)
    # 预分配内存
    r2_list = np.zeros(36)
    # 并行计算R2分数
    r2_list = Parallel(n_jobs=-1)(delayed(r2_score)(y_test[:, i], y_test_predicted[:, i]) for i in range(36))
    return np.array(r2_list).mean()

def LHS_LHS_normal_uniform(p):
    # ! 读取参数
    parameter_range_0 = p[0]
    parameter_range_1 = p[1]
    parameter_quantity = int(p[2])

    # 第一阶段是 LHS 毋庸置疑
    x_temp_1, y_temp_1 = sampling_LHS(parameter_number, 1024-parameter_quantity, _range=np.array([-3, 3]))
    # x_temp_2, y_temp_2 = sampling_LHS(parameter_number, parameter_quantity, _range=np.array([parameter_range_0, parameter_range_1]))
    # x_temp_2, y_temp_2 = sampling_normal(parameter_number, parameter_quantity)
    x_temp_2, y_temp_2 = sampling_uniform(parameter_number, parameter_quantity, _range=np.array([parameter_range_0, parameter_range_1]))

    # ！ 合并两阶段的数据
    x_train, y_train = np.vstack((x_temp_1, x_temp_2)), np.vstack((y_temp_1, y_temp_2))

    # ！ 开始训练
    sm = KPLS(theta0=[1e-2], print_global=False)
    sm.set_training_values(x_train, y_train)
    sm.train()
    return 1 - test_acquired(sm, x_test=x_test, y_test=y_test)

range_total_0 = np.arange(-2.5, -1.5, 0.15)
range_total_1 = np.arange(1.5, 2.5, 0.15)
quantity_total = np.arange(100, 550, 80)
print(range_total_0.shape)
print(range_total_1.shape)
print(quantity_total.shape)

start_time = time.time()
print("开始进行所有网格搜索计算 ....")

ranges_0, ranges_1, quantities = np.meshgrid(range_total_0, range_total_1, quantity_total)
params = np.stack([ranges_0.ravel(), ranges_1.ravel(), quantities.ravel()], axis=-1) # 将 ranges 和 quantities 展平，并将它们堆叠成一个二维数组
results = np.array([LHS_LHS_normal_uniform(p=param) for param in params]) # 对每一组参数调用 LHS_LHS_normal_uniform 函数，并将结果保存在 results 中
results_total = np.hstack([params, results[:, None]]) # 将 params 和 results 拼接成最终的 results_total 数组

end_time = time.time()
print(f"共计耗时: {((end_time - start_time)/60):.3f} min")

np.savetxt('./GA_results/results_total_C.txt', results_total)