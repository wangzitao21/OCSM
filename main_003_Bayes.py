from sklearn.manifold import TSNE
import numpy as np
import pickle
from bayes_opt import BayesianOptimization
import time
import matplotlib.pyplot as plt
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from smt.surrogate_models import KRG, KPLS, KPLSK
from sklearn.metrics import r2_score
np.set_printoptions(suppress=True)

from methods.MCMCmodel import metropolis_hastings, genex
from utils.surrogate_generator import sampling_LHS, sampling_normal, sampling_uniform, sampling_random, sampling_factorial

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import pickle

from joblib import Parallel, delayed

############## 代理模型 ################
print('开始训练代理模型 ...')
data = np.loadtxt('./GA_results/results_total_A.txt')
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

############## 贝叶斯优化 ################
# def target(p1, p2, p3):
#     p = np.array([[p1, p2, p3],])
#     return 1 - sm.predict_values(p)[0,0]

# bo = BayesianOptimization(
#     target,
#     {'p1': (-2.5, -1.5),
#      'p2': (1.5, 2.5),
#      'p3': (100, 550),
#     }
# )

# logger = JSONLogger(path="./Bayes_results/Bayes_C.json")
# bo.subscribe(Events.OPTIMIZATION_STEP, logger)

# start_time = time.time()
# print("开始进行贝叶斯优化 ....")
# bo.maximize(
#     init_points=2000,
#     n_iter=1000,
# )
# end_time = time.time()
# print(f"贝叶斯优化共计耗时: {((end_time - start_time)/60):.3f} min")
# print(bo.max)

# MCMC #############################################
def target(p):
    p[-1] = np.round(p[-1])
    
    # p = np.array([p,])
    p = p.reshape(-1, 3)
    print(p)
    print(1 - np.array([sm.predict_values(p)[0,0],]))
    return 1 - np.array([sm.predict_values(p)[0,0],])

N1 = 1
Npar = 3 # 未知参数数量
# A
range_A = np.array([[-2.6, -1.85],
                    [1.85, 2.6],
                    [410, 700]
                   ])
# B
range_B = np.array([[-3.0, -2.0],
                    [2.0, 3.0],
                    [100, 400]
                   ])
# C
range_C = np.array([[-2.5, -1.5],
                   [1.5, 2.5],
                   [100, 550]
                  ])

range_ = range_A

# sd = np.ones(3) * 0.01
sd = np.array([0.01, 0.01, 1])
x1 = genex(range_, N1).reshape(-1)

q = 800000 # 迭代次数
obs = np.array([1.0, ])

start_time = time.time()
accepted, rejected, scores = metropolis_hastings(x1, obs, sd, q, range_, target)
end_time = time.time()
print(f"MCMC 耗时: {((end_time - start_time)/60):.3f} min")
print("接受了 {} 个样品".format(accepted.shape[0]))

np.save('./Bayes_results/accepted_A.npy', accepted)
np.save('./Bayes_results/rejected_A.npy', rejected)
np.save('./Bayes_results/scores_A.npy', scores)


















