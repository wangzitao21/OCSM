import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from smt.applications import MOE
from smt.surrogate_models import KRG, KPLS, KPLSK
from sklearn.metrics import r2_score
np.set_printoptions(suppress=True)
from utils.surrogate_generator import sampling_LHS, sampling_normal, sampling_uniform, sampling_random, sampling_factorial
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

def test_acquired(sm, x_test=x_test, y_test=y_test):
    y_test_predicted = sm.predict_values(x_test)
    # 预分配内存
    r2_list = np.zeros(36)
    # 并行计算R2分数
    r2_list = Parallel(n_jobs=-1)(delayed(r2_score)(y_test[:, i], y_test_predicted[:, i]) for i in range(36))
    return np.array(r2_list).mean()

# ! ######################################################################
def LHS():
    x_train, y_train = sampling_LHS(parameter_number, 1024, _range=np.array([-3, 3]))
    # sm = KRG(theta0=[1e-2], print_global=False) # ! 开始训练
    # sm = MOE(smooth_recombination=False, n_clusters=36, allow=["KRG", "LS", "IDW"])
    sm = KPLS(theta0=[1e-2], print_global=False)
    sm.set_training_values(x_train, y_train)
    sm.train()

    with open("./smt_sm/LHS.pkl", "wb") as f:
       pickle.dump(sm, f)  
    
    return test_acquired(sm, x_test=x_test, y_test=y_test)
    
def LHS_LHS(p):
    # ! 读取参数
    parameter_range_0 = p[0]
    parameter_range_1 = p[1]
    parameter_quantity = int(p[2])

    x_temp_1, y_temp_1 = sampling_LHS(parameter_number, 1024-parameter_quantity, _range=np.array([-3, 3]))
    x_temp_2, y_temp_2 = sampling_LHS(parameter_number, parameter_quantity, _range=np.array([parameter_range_0, parameter_range_1]))
    x_train, y_train = np.vstack((x_temp_1, x_temp_2)), np.vstack((y_temp_1, y_temp_2)) # ! 合并两阶段的数据
    sm = KRG(theta0=[1e-2], print_global=False) # ! 开始训练
    # sm = KPLS(theta0=[1e-2], print_global=False)
    sm.set_training_values(x_train, y_train)
    sm.train()

    with open("./smt_sm/LHS_LHS.pkl", "wb") as f:
       pickle.dump(sm, f)  
    
    return test_acquired(sm, x_test=x_test, y_test=y_test)

def LHS_normal(p):
    # ! 读取参数
    parameter_range_0 = p[0]
    parameter_range_1 = p[1]
    parameter_quantity = int(p[2])

    x_temp_1, y_temp_1 = sampling_LHS(parameter_number, 1024-parameter_quantity, _range=np.array([parameter_range_0, parameter_range_1]))
    x_temp_2, y_temp_2 = sampling_normal(parameter_number, parameter_quantity)
    x_train, y_train = np.vstack((x_temp_1, x_temp_2)), np.vstack((y_temp_1, y_temp_2)) # ! 合并两阶段的数据
    sm = KRG(theta0=[1e-2], print_global=False) # ! 开始训练
    # sm = KPLS(theta0=[1e-2], print_global=False)
    sm.set_training_values(x_train, y_train)
    sm.train()

    with open("./smt_sm/LHS_normal.pkl", "wb") as f:
       pickle.dump(sm, f)    
    
    return test_acquired(sm, x_test=x_test, y_test=y_test)

def LHS_uniform(p):
    # ! 读取参数
    parameter_range_0 = p[0]
    parameter_range_1 = p[1]
    parameter_quantity = int(p[2])

    x_temp_1, y_temp_1 = sampling_LHS(parameter_number, 1024-parameter_quantity, _range=np.array([-3, 3]))
    x_temp_2, y_temp_2 = sampling_uniform(parameter_number, parameter_quantity, _range=np.array([parameter_range_0, parameter_range_1]))
    x_train, y_train = np.vstack((x_temp_1, x_temp_2)), np.vstack((y_temp_1, y_temp_2)) # ! 合并两阶段的数据
    sm = KRG(theta0=[1e-2], print_global=False) # ! 开始训练
    # sm = KPLS(theta0=[1e-2], print_global=False)
    sm.set_training_values(x_train, y_train)
    sm.train()

    with open("./smt_sm/LHS_uniform.pkl", "wb") as f:
       pickle.dump(sm, f)

    return test_acquired(sm, x_test=x_test, y_test=y_test)
# ! ######################################################################

# LHS_scores = LHS()
LHS_LHS_scores = LHS_LHS(np.array([-1.8684347, 2.44848516, 528.262]))
# LHS_normal_scores = LHS_normal(np.array([-2.27418499, 2.2963877, 180]))
# LHS_uniform_scores = LHS_uniform(np.array([-1.54100477, 2.2576027, 426]))

# print('LHS scores: ', LHS_scores)
print('LHS-LHS scores: ', LHS_LHS_scores)
# print('LHS-normal scores: ', LHS_normal_scores)
# print('LHS-uniform scores: ', LHS_uniform_scores)
