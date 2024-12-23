import numpy as np
import pickle

# 导入保存的模型
sm = None
filename = "./surrogate_data/LHS_LHS.pkl"
with open(filename, "rb") as f:
    sm = pickle.load(f)

# x_suiji = np.random.uniform(low=-2.5, high=2.5, size=48)
# indices = np.array([45, 43, 40, 37, 46, 33, 31, 39, 42, 34, 36, 27, 28, 30, 19, 25, 10,
#         15, 16, 22, 13, 18,  4,  3, 21, 24,  7,  6,  9, 12,  1,  0, 47, 41,
#         44, 35, 38, 29, 32, 26, 20, 17, 23,  8,  5, 11, 14,  2])
# parameter_number = 39
# indices = indices[:parameter_number]

# 任意实例
def forwardmodel(x):
    # x = x.reshape(-1) # (30,)
    # for i in range(parameter_number):
      # x_suiji[indices[i]] = x[i]
    # x = x_suiji

    x = x.reshape(1, -1)
    return sm.predict_values(x).reshape(-1)