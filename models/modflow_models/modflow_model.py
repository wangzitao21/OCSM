import os
import uuid
import flopy
import numpy as np

from utils.utils import get_timeseries, get_data_timeseries, sgs
from utils.utils import deleteFilesAndFolders

# ! 加载 KLE 参数
nx, ny, mean_logk, lamda_xy, fn_x, fn_y, _ = np.load('./data/KLE34.npy', allow_pickle=True)
def calculate_logk(nx, ny, mean_logk, lamda_xy, fn_x, fn_y, kesi):
    # print(kesi.shape)
    kesi = kesi.reshape(1, -1) # 增维, 必须
    logk = np.zeros((nx, ny))
    # 由随机数计算渗透率场
    for i_x in range(nx):
        for i_y in range(ny):
            logk[i_y, i_x] = mean_logk + np.sum(np.sqrt(lamda_xy) * fn_x[i_x][0] * \
                                    fn_y[i_y][0] * kesi.transpose())
    return logk

# ! K 值范围
targetMin_K, targetMax_K = 50, 300

# ! 不变参数
sim_name = 'simulation'
gwfname = 'gwf_model'

# Units
length_units = "meters"
time_units = "days"

icelltype = 1

nouter, ninner = 100, 300
hclose, rclose, relax = 1e-6, 1e-6, 1.0

# Discretization
nlay = 1 # ! 单层
nrow = 64
ncol = 64
delr = 200.0
delc = 200.0
top = 2678 # np.loadtxt('top.txt') # ! 修改
botm = 2628.0 # ! 修改
idomain = 1

ss = 0.000005 # sgs([0.000001, 0.00001], seed=263, len_scale=8, gridX=64, gridY=64)
sy = 0.1
strt = 2678.0

east_flow = np.loadtxt('./data/east.csv', delimiter=',', encoding='utf-8-sig')
east_spd = []
for i in np.arange(east_flow.shape[0]):
    east_spd.append([tuple(east_flow[i, :3].astype(int)-1), 2676.8])
# east_ts_file = os.path.join("./data/east_ts.csv")
# east_ts = get_timeseries(east_ts_file, 'east_flow', "linear")
east_spd = {0: east_spd}

west_flow = np.loadtxt('./data/west.csv', delimiter=',', encoding='utf-8-sig')
west_spd = []
for i in np.arange(west_flow.shape[0]):
    west_spd.append([tuple(west_flow[i, :3].astype(int)-1), 'west_flow'])
west_ts_file = os.path.join("./data/west_ts.csv")
west_ts = get_timeseries(west_ts_file, 'west_flow', "linear")
west_spd = {0: west_spd}

bs_location = np.loadtxt('./data/bs.csv', delimiter=',', encoding='utf-8-sig')
bs_spd = []
for i in np.arange(bs_location.shape[0]):
    bs_spd.append([tuple(bs_location[i, :3].astype(int)-1), 'bs_stage', 6000, 2660]) # , 305])
bs_stage_file = os.path.join("./data/bs_ts.csv")
bs_ts = get_timeseries(bs_stage_file, 'bs_stage', "linear")
bs_spd = {0: bs_spd}

x = np.array([-2450, -2960, -342.6, -1100, -900, -544, -2109, -943, -20, -2800, 0])

cl_location = np.loadtxt('./data/cl.csv', delimiter=',', encoding='utf-8-sig')
cl_spd = []
for i in np.arange(cl_location.shape[0]):
    cl_spd.append([tuple(cl_location[i, :3].astype(int)-1), 'chd_flow_1']) # , 200,]) # 350.0]) # 2678-cl_location[i, 5], 
cl_ts_data = [(i, x[i]) for i in range(11)] # ? 待反演的参数
cl_ts_1 = get_data_timeseries(cl_ts_data, 'chd_flow_1', "linear")
cl_spd = {0: cl_spd}

obs_location = np.loadtxt('./data/OBS.csv', delimiter=',', encoding='utf-8-sig')

def forwardmodel(kesi):

    # nx, ny, mean_logk, lamda_xy, fn_x, fn_y, _ = np.load('./data/KLE34.npy', allow_pickle=True)
    logk11 = calculate_logk(nx, ny, mean_logk, lamda_xy, fn_x, fn_y, kesi)
    k11 = np.exp(logk11)

    # targetMin_K, targetMax_K = 50, 300
    k11 = np.interp(k11, (k11.min(), k11.max()), (targetMin_K, targetMax_K))

    # Filenames
    sim_ws = os.path.join('simulation_folder', str(uuid.uuid4()))
    
    # icelltype = 1
    # sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name='mf6', verbosity_level=0)

    sim = flopy.mf6.MFSimulation(sim_name=sim_name, sim_ws=sim_ws, exe_name='../modflow/mf6', verbosity_level=0)

    # Temporal discretization
    nper = 1 # 2
    perlen  = [1.0] # , 5.0]
    nstp = [1] # , 5]
    tsmult = [1.0] # , 1.0]
    tdis_ds = list(zip(perlen, nstp, tsmult))
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_ds, time_units=time_units)

    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True, model_nam_file="{}.nam".format(gwfname))
    # Solver parameters
    # nouter, ninner = 100, 300
    # hclose, rclose, relax = 1e-6, 1e-6, 1.0
    imsgwf = flopy.mf6.ModflowIms(sim, 
                                  complexity="COMPLEX",
                                  print_option="SUMMARY",
                                  outer_dvclose=hclose,
                                  outer_maximum=nouter,
                                  under_relaxation="NONE",
                                  inner_maximum=ninner,
                                  inner_dvclose=hclose,
                                  rcloserecord=rclose,
                                  linear_acceleration="CG",
                                  scaling_method="NONE",
                                  reordering_method="NONE",
                                  relaxation_factor=relax,
                                  filename="{}.ims".format(gwfname)
                                 )
    sim.register_ims_package(imsgwf, [gwf.name])
    flopy.mf6.ModflowGwfdis(gwf, length_units=length_units, nlay=nlay, nrow=nrow, ncol=ncol, delr=delr, delc=delc,
                        top=top, botm=botm, idomain=idomain, filename="{}.dis".format(gwfname))
    k33 = k11 * 0.6
    flopy.mf6.ModflowGwfnpf(gwf, save_flows=False, icelltype=icelltype, k=k11, k33=k33,
                            save_specific_discharge=True,
                            # tvk_filerecord=gwfname+'.tvk',
                            filename="{}.npf".format(gwfname),)
    flopy.mf6.ModflowGwfsto(gwf, ss=ss, sy=sy)
    flopy.mf6.ModflowGwfic(gwf, strt=strt, filename="{}.ic".format(gwfname))

    # # chd_flow_value = 2676.0
    # chd_flow = np.loadtxt('./data/north_lake.csv', delimiter=',', encoding='utf-8-sig')
    # chd_spd = []
    # for i in np.arange(chd_flow.shape[0]):
    #     chd_spd.append([tuple(chd_flow[i, :3].astype(int)-1), 'chd_flow']) # chd_flow_value

    # chd_ts_file = os.path.join("./data/north_lake_ts.csv")
    # chd_ts = get_timeseries(chd_ts_file, 'chd_flow', "linear")
    # chd_spd = {0: chd_spd}
    # flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd_spd, save_flows=False, timeseries=chd_ts, pname="NHS", # auxiliary=["TDS",],# 'K'],
    #                         filename="{}.nhs.chd".format(gwfname),)

    chd_flow = np.loadtxt('./data/south.csv', delimiter=',', encoding='utf-8-sig')
    chd_spd = []
    for i in np.arange(chd_flow.shape[0]):
        chd_spd.append([tuple(chd_flow[i, :3].astype(int)-1), 'chd_flow']) # chd_flow_value

    chd_ts_file = os.path.join("./data/south_ts.csv")
    chd_ts = get_timeseries(chd_ts_file, 'chd_flow', "linear")
    chd_spd = {0: chd_spd}
    flopy.mf6.ModflowGwfchd(gwf,
                            # maxbound=len(chd_spd),
                            stress_period_data=chd_spd,
                            save_flows=False,
                            timeseries=chd_ts,
                            pname="SHS",
                            # auxiliary=["TDS",],# 'K'],
                            filename="{}.shs.chd".format(gwfname),
                            )
############################ ! 东部边界 ############################
    flopy.mf6.ModflowGwfchd(gwf,
                            # maxbound=len(east_spd),
                            stress_period_data=east_spd,
                            save_flows=False,
                            # timeseries=east_ts,
                            pname="EAST",
                            # auxiliary=["TDS",],# 'K'],
                            filename="{}.east.chd".format(gwfname),
                            )
############################ ! 西部边界 ############################

    flopy.mf6.ModflowGwfchd(gwf,
                            # maxbound=len(chd_spd),
                            stress_period_data=west_spd,
                            save_flows=False,
                            timeseries=west_ts,
                            pname="WEST",
                            # auxiliary=["TDS",],# 'K'],
                            filename="{}.west.chd".format(gwfname),
                            )

############################ ! 采卤渠水头边界 ############################
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=cl_spd,
                        timeseries=cl_ts_1, pname="CL-1", filename="{}.cl_1.wel".format(gwfname),)

############################ ! 补水渠水头边界 ############################

    flopy.mf6.ModflowGwfriv(gwf,
                            stress_period_data=bs_spd,
                            timeseries=bs_ts,
                            pname="BS",
                            # auxiliary=["TDS",],
                            filename="{}.bs.riv".format(gwfname),
                            )
    
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="{}.hds".format(gwfname), budget_filerecord="{}.cbc".format(gwfname),
                                budgetcsv_filerecord="{}.oc.csv".format(gwfname), saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
                                printrecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],)

    sim.write_simulation() # silent=True
    success, _ = sim.run_simulation()
    if not success:
        raise Exception("MODFLOW 6 did not terminate normally.")

    head = gwf.oc.output.head().get_alldata()

    headTotal = []
    for row in obs_location[:, 1:]:
        # print(row)
        i = int(row[0]) - 1
        j = int(row[1]) - 1
        headTotal.append(head[:, 0, i, j].ravel())

    deleteFilesAndFolders(sim_ws)

    return np.concatenate(headTotal)# [::18]
    # return head