import os
import shutil
import numpy as np
import gstools as gs

# 生成均匀分布的随机数，用于初始 x 生成
def genex(range_, N=1):
    Npar = range_.shape[0]
    x = np.empty((Npar, N))
    x[:] = np.nan
    for i in range(N):
        x[:,i] = range_[:,0] + (range_[:,1] - range_[:,0]) * np.random.rand(Npar)
    return x

# 生成样品
def sgs(target, seed, len_scale, gridX, gridY):
    x = np.arange(gridX)
    y = np.arange(gridY)

    model = gs.Gaussian(dim=2, var=1, len_scale=len_scale)
    srf = gs.SRF(model=model, seed=seed)

    srf.structured([x, y])
    field = srf.transform("normal_to_arcsin")

    targetMin, targetMax = target[0], target[1]
    field = np.interp(field, (field.min(), field.max()), (targetMin, targetMax))

    # plt.imshow(field.T, origin='lower', cmap="rainbow")
    # plt.colorbar()
    return field.T

# 删除 simulation_folder 子目录及文件
def deleteFilesAndFolders(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# modflow 获取时间序列, 用于边界条件
def get_timeseries(fname, names, interpolation, filename=None):
    tsdata = []
    for row in np.genfromtxt(fname, delimiter=",", comments="#"):
        tsdata.append(tuple(row))
    tsdict = {"timeseries": tsdata,
              "time_series_namerecord": names,
              "interpolation_methodrecord": interpolation,
              }
    if filename is not None:
        tsdict["filename"] = filename
    return tsdict

# 与上同, 但接受非表格文件
def get_data_timeseries(tsdata, names, interpolation, filename=None):
    tsdict = {"timeseries": tsdata,
              "time_series_namerecord": names,
              "interpolation_methodrecord": interpolation,
              }
    if filename is not None:
        tsdict["filename"] = filename
    return tsdict
