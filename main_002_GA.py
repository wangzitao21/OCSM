from sko.GA import GA

import os
import time
import pickle
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

from sko.tools import set_run_mode

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

# def test_acquired(sm, x_test=x_test, y_test=y_test):
#     y_test_predicted = sm.predict_values(x_test)
#     # 预分配内存
#     r2_list = np.zeros(36)
#     # 顺序计算R2分数
#     for i in range(36):
#         r2_list[i] = r2_score(y_test[:, i], y_test_predicted[:, i])
#     return np.array(r2_list).mean()

############## 代理模型 ################
print('开始训练代理模型 ...')
data = np.loadtxt('./GA_results/results_total_C.txt')
data.shape

xt = data[:, :3]
yt = data[:, -1]

sm = KRG()
sm.set_training_values(xt, yt)
sm.train()
# y_pred = sm.predict_values(xt_test)

# 保存模型
# filename = "surrogate.pkl"
# with open(filename, "wb") as f:
#    pickle.dump(sm, f)

# 加载模型
# sm = None
# filename = "surrogate.pkl"
# with open(filename, "rb") as f:
#    sm = pickle.load(f)
    
print('代理模型训练完成 ...\n\n')

############## 遗传算法 ################
def surrogate_model(p):
    p = np.array([p,])
    return sm.predict_values(p)[0,0]

set_run_mode(surrogate_model, mode='multiprocessing')
# A
# ga = GA(func=surrogate_model, 
#         n_dim=3,
#         size_pop=300, # 种群数
#         max_iter=30, # 迭代次数 
#         prob_mut=0.001, # 变异概率
#         # lb=[-2.6, 1.85, 410], # 最小值
#         # ub=[-1.85, 2.6, 700], # 最大值
#         # lb=[-2.6, 1.85, 410], # 最小值
#         # ub=[-1.85, 2.6, 700], # 最大值
#         lb=[-2.5, 1.5, 100], # 最小值
#         ub=[-1.5, 2.5, 550], # 最大值
#         precision=[1e-7, 1e-7, 5],
#         # constraint_eq=[lambda x: x[0]+x[1]+x[2]+x[3]-1024] # 线性约束
#        )
# B
# ga = GA(func=surrogate_model, 
#         n_dim=3,
#         size_pop=300, # 种群数
#         max_iter=30, # 迭代次数 
#         prob_mut=0.001, # 变异概率
#         lb=[-3.0, 2.0, 100], # 最小值
#         ub=[-2.0, 3.0, 400], # 最大值
#         precision=[1e-7, 1e-7, 5],
#         # constraint_eq=[lambda x: x[0]+x[1]+x[2]+x[3]-1024] # 线性约束
#        )
# C
ga = GA(func=surrogate_model, 
        n_dim=3,
        size_pop=300, # 种群数
        max_iter=30, # 迭代次数 
        prob_mut=0.001, # 变异概率
        lb=[-2.5, 1.5, 100], # 最小值
        ub=[-1.5, 2.5, 550], # 最大值
        precision=[1e-7, 1e-7, 5],
        # constraint_eq=[lambda x: x[0]+x[1]+x[2]+x[3]-1024] # 线性约束
       )

start_time = time.time()
print("开始进行遗传算法计算 ....")
best_x, best_y = ga.run()
end_time = time.time()
print(f"遗传算法共计耗时: {((end_time - start_time)/60):.3f} min")

Y_history = ga.all_history_Y # 计算每代所有样品的 y
Y_best_history = ga.generation_best_Y # 计算每代最佳样品的 y
np.save('./GA_results/GA_Y_C.npy', Y_history)
np.save('./GA_results/GA_Y_best_C.npy', Y_best_history)

print('best_x:', best_x, '\n', 'best_y:', best_y)

# print("history: ", ga.all_history_Y)