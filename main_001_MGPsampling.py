from sko.GA import GA

import os
import time
import numpy as np

from smt.surrogate_models import KRG, KPLS, KPLSK
from sklearn.metrics import r2_score
np.set_printoptions(suppress=True)
from utils.surrogate_generator import sampling_LHS, sampling_normal, sampling_uniform, sampling_random

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

# def LHS_LHS_normal_uniform(p):
#     # ! 读取参数
#     parameter_range_0 = p[0]
#     parameter_range_1 = p[1]
#     parameter_quantity = int(p[2])

#     # 第一阶段是 LHS 毋庸置疑
#     x_temp_1, y_temp_1 = sampling_LHS(parameter_number, 1024-parameter_quantity, _range=np.array([-3, 3]))
#     x_temp_2, y_temp_2 = sampling_LHS(parameter_number, parameter_quantity, _range=np.array([parameter_range_0, parameter_range_1]))

#     #     x_temp_2, y_temp_2 = sampling_normal(parameter_number, parameter_quantity)
#     #     x_temp_2, y_temp_2 = sampling_uniform(parameter_number, parameter_quantity, _range=parameter_range)

#     # ！ 合并两阶段的数据
#     x_train, y_train = np.vstack((x_temp_1, x_temp_2)), np.vstack((y_temp_1, y_temp_2))

#     # ！ 开始训练
#     sm = KPLS(theta0=[1e-2], print_global=False)
#     sm.set_training_values(x_train, y_train)
#     sm.train()
#     return test_acquired(sm, x_test=x_test, y_test=y_test)

def LHS_LHS_normal_uniform(p):
    # ! 读取参数
    parameter_range = p[0]
    parameter_quantity = int(p[1])

    # 第一阶段是 LHS 毋庸置疑 np.array([-3.0, 3.0])
    x_temp_1, y_temp_1 = sampling_LHS(parameter_number, 1024-parameter_quantity, _range=np.array([parameter_range*(-1), parameter_range]))
    # x_temp_2, y_temp_2 = sampling_LHS(parameter_number, parameter_quantity, _range=np.array([parameter_range*(-1), parameter_range]))

    x_temp_2, y_temp_2 = sampling_normal(parameter_number, parameter_quantity)
    # x_temp_2, y_temp_2 = sampling_uniform(parameter_number, parameter_quantity, _range=np.array([parameter_range*(-1), parameter_range]))

    # ！ 合并两阶段的数据
    x_train, y_train = np.vstack((x_temp_1, x_temp_2)), np.vstack((y_temp_1, y_temp_2))

    # ！ 开始训练
    sm = KPLS(theta0=[1e-2], print_global=False)
    sm.set_training_values(x_train, y_train)
    sm.train()
    return 1 - test_acquired(sm, x_test=x_test, y_test=y_test)

############## 遗传算法 ################
# range_total_0 = np.arange(-2.75, -1.5, 0.5)
# range_total = np.arange(1.5, 2.75, 0.1) # np.arange(1.5, 2.75, 0.2)
# quantity_total =  np.arange(400, 850, 20) # np.arange(400, 850, 30)

range_total = np.arange(2.0, 3.0, 0.1) # np.arange(1.5, 2.75, 0.2)
quantity_total =  np.arange(100, 400, 20) # np.arange(400, 850, 30)

start_time = time.time()
print("开始进行所有网格搜索计算 ....")

ranges, quantities = np.meshgrid(range_total, quantity_total) # 创建一个网格，其中包含所有可能的 range_single 和 quantity_single 组合
params = np.stack([ranges.ravel(), quantities.ravel()], axis=-1) # 将 ranges 和 quantities 展平，并将它们堆叠成一个二维数组
results = np.array([LHS_LHS_normal_uniform(p=param) for param in params]) # 对每一组参数调用 LHS_LHS_normal_uniform 函数，并将结果保存在 results 中
results_total = np.hstack([params, results[:, None]]) # 将 params 和 results 拼接成最终的 results_total 数组

end_time = time.time()
print(f"共计耗时: {((end_time - start_time)/60):.3f} min")

np.savetxt('./GA_results/MGPsampling_scenarioB.txt', results_total)

