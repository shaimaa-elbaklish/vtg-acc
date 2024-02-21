import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import savemat

from acc_model import ACCModelConfig
from vtg_acc_controller import VTG_ACC_Config
from vtg_smc_controller import VTG_SMC_Config
from utils_plotting import plot_leader_multiple_followers


FILES_ACC_MAX = ['ASta_040719_platoon4']
FILES_ACC_MIN = [
    'ASta_050719_platoon1', 'ASta_050719_platoon2', 'ASta_040719_platoon5', 'ASta_040719_platoon6', 
    'ASta_040719_platoon7', 'ASta_040719_platoon8', 'ASta_040719_platoon9'
]
FILES_ACC_HUMAN = ['ASta_040719_platoon3', 'ASta_040719_platoon10']
ACC_IDX = 0


df = pd.read_csv(f'./openacc_data/{FILES_ACC_MIN[ACC_IDX]}.csv', skiprows=5)
df_note = pd.read_csv(f'./openacc_data/{FILES_ACC_MIN[ACC_IDX]}.csv', delimiter=';', nrows=5, header=None)

num_veh = int(df_note.loc[2,0].split(',')[1])
veh_types = df_note.loc[1,0].split(',')
veh_types = [x for x in veh_types if x != '']
veh_types = veh_types[1:]
freq = 10
for c in range(1, num_veh):
    df[f"posDiff{c}"] = np.sqrt((df[f"E{c}"] - df[f"E{c+1}"])**2 + (df[f"N{c}"] - df[f"N{c+1}"])**2 + (df[f"U{c}"] - df[f"U{c+1}"])**2)
    df[f"acceleration{c+1}"] = df[f"Speed{c+1}"].diff().shift(-1).fillna(0) * freq
    df[f"timeHeadway{c}"] = df[f"posDiff{c}"] / df[f"Speed{c+1}"]
    df[f"timeGap{c}"] = df[f"IVS{c}"] / df[f"Speed{c+1}"]
df["Time"] = df["Time"] - df.loc[0, "Time"]
start = len(df['Time']) // 20
stop = (len(df['Time']) * 19) // 20

Time = df['Time'].to_numpy()[start:stop]
Leader_Velocity = df['Speed1'].to_numpy()[start:stop]

acc_config = ACCModelConfig(
    k0 = 0.10,
    k1 = 0.23,
    k2 = 0.07,
    time_gap = 1.2,
    desired_speed = 30,
    still_distance = 3.0,
    vehicle_length = 5.0,
    use_acceleration_constraints = True,
    use_MFC_constraints = False
)
acc_followers = [
    acc_config.create(f"./calibration_results/{veh_types[i]}_best_fit_parameters.json") for i in range(1, num_veh)
]

vtg_acc_config = VTG_ACC_Config(
    k0 = 0.10,
    k1 = 0.23,
    k2 = 0.07,
    desired_time_gap = 1.2,
    desired_speed = 30,
    still_distance = 3.0,
    vehicle_length = 5.0,
    use_acceleration_constraints = True,
    use_MFC_constraints = False,
    hinf_bound = 0.9
)
vtg_acc_followers = [
    vtg_acc_config.create(f"./calibration_results/{veh_types[i]}_best_fit_parameters.json") for i in range(1, num_veh)
]

vtg_smc_config = VTG_SMC_Config(
    k0 = 0.10,
    desired_speed = 30,
    lam = 2.0,
    T = 0.0019,
    G = 0.0448,
    still_distance = 3.0,
    vehicle_length = 5.0,
    use_acceleration_constraints = True,
    use_MFC_constraints = False
)
vtg_smc_followers = [
    vtg_smc_config.create(f"./calibration_results/{veh_types[i]}_best_fit_parameters.json") for i in range(1, num_veh)
]

T = Time.shape[0]
ctg_vel_arr = np.zeros(shape=(num_veh,T))
ctg_space_headway_arr = np.zeros(shape=(num_veh-1,T))
ctg_ego_acc_arr = np.zeros(shape=(num_veh-1,T-1))
ctg_time_gap_arr = np.zeros(shape=(num_veh-1,T-1))

vtg_vel_arr = np.zeros(shape=(num_veh,T))
vtg_space_headway_arr = np.zeros(shape=(num_veh-1,T))
vtg_ego_acc_arr = np.zeros(shape=(num_veh-1,T-1))
vtg_time_gap_arr = np.zeros(shape=(num_veh-1,T-1))

smc_vel_arr = np.zeros(shape=(num_veh,T))
smc_space_headway_arr = np.zeros(shape=(num_veh-1,T))
smc_ego_acc_arr = np.zeros(shape=(num_veh-1,T-1))
smc_time_gap_arr = np.zeros(shape=(num_veh-1,T-1))

# Initialization
ctg_vel_arr[0, :] = Leader_Velocity
for v in range(num_veh-1):
    ctg_vel_arr[v+1, 0] = df.loc[start, f'Speed{v+2}']
    ctg_space_headway_arr[v, 0] = df.loc[start, f'posDiff{v+1}']
    ctg_time_gap_arr[v, :] = acc_followers[v]._tg * np.ones(shape=(T-1,))

vtg_vel_arr[0, :] = Leader_Velocity
for v in range(num_veh-1):
    vtg_vel_arr[v+1, 0] = df.loc[start, f'Speed{v+2}']
    vtg_space_headway_arr[v, 0] = df.loc[start, f'posDiff{v+1}']

smc_vel_arr[0, :] = Leader_Velocity
for v in range(num_veh-1):
    smc_vel_arr[v+1, 0] = df.loc[start, f'Speed{v+2}']
    smc_space_headway_arr[v, 0] = df.loc[start, f'posDiff{v+1}']

time_window = 60
# Simulation Loop
for t in range(T-1):
    for v in range(num_veh-1):
        # ACC
        ctg_ego_acc_arr[v, t] = acc_followers[v].get_acceleration_command(
            ctg_vel_arr[v, t], ctg_vel_arr[v+1, t], ctg_space_headway_arr[v, t]
        )
        ctg_space_headway_arr[v, t+1], ctg_vel_arr[v+1, t+1] = acc_followers[v].step(
            ctg_vel_arr[v, t], ctg_vel_arr[v+1, t], ctg_space_headway_arr[v, t], ctg_ego_acc_arr[v, t]
        )

        # VTG ACC
        # v_eq = np.median(vtg_vel_arr[v, max(0, t-time_window):t+1])
        vtg_time_gap_arr[v, t], vtg_ego_acc_arr[v, t] = vtg_acc_followers[v].get_acceleration_command(
            vtg_vel_arr[v, t], vtg_vel_arr[v+1, t], vtg_space_headway_arr[v, t],
            # vel_eq=v_eq, compute_P=(Time[t] % time_window == 0)
        )
        vtg_space_headway_arr[v, t+1], vtg_vel_arr[v+1, t+1] = vtg_acc_followers[v].step(
            vtg_vel_arr[v, t], vtg_vel_arr[v+1, t], vtg_space_headway_arr[v, t], vtg_ego_acc_arr[v, t]
        )
        
        # VTG SMC
        smc_ego_acc_arr[v, t] = vtg_smc_followers[v].get_acceleration_command(
            smc_vel_arr[v, t], smc_vel_arr[v+1, t], smc_space_headway_arr[v, t]
        )
        smc_space_headway_arr[v, t+1], smc_vel_arr[v+1, t+1] = vtg_smc_followers[v].step(
            smc_vel_arr[v, t], smc_vel_arr[v+1, t], smc_space_headway_arr[v, t], smc_ego_acc_arr[v, t]
        )

plot_leader_multiple_followers(
    Time, Leader_Velocity, ctg_vel_arr[1:, :], ctg_space_headway_arr, ctg_ego_acc_arr, 
    ctg_space_headway_arr / ctg_vel_arr[1:, :], 
    fig_title="Constant Time Gap ACC Followers"
)

plot_leader_multiple_followers(
    Time, Leader_Velocity, vtg_vel_arr[1:, :], vtg_space_headway_arr, vtg_ego_acc_arr, 
    vtg_space_headway_arr / vtg_vel_arr[1:, :], 
    fig_title="Variable Time Gap ACC Followers"
)

plot_leader_multiple_followers(
    Time, Leader_Velocity, smc_vel_arr[1:, :], smc_space_headway_arr, smc_ego_acc_arr, 
    smc_space_headway_arr / smc_vel_arr[1:, :], 
    fig_title="VTG SMC Followers"
)

plot_leader_multiple_followers(
    df['Time'].to_numpy()[start:stop], df['Speed1'].to_numpy()[start:stop], 
    (df[['Speed2', 'Speed3', 'Speed4', 'Speed5']].to_numpy().T)[:, start:stop],
    (df[['posDiff1', 'posDiff2', 'posDiff3', 'posDiff4']].to_numpy().T)[:, start:stop], 
    (df[['acceleration2', 'acceleration3', 'acceleration4', 'acceleration5']].to_numpy().T)[:, start:stop-1],
    (df[['timeHeadway1', 'timeHeadway2', 'timeHeadway3', 'timeHeadway4']].to_numpy().T)[:, start:stop],
    fig_title="openACC Data"
)

plt.show()

mat_results = {
    'num_followers': num_veh - 1,
    'Time': df['Time'].to_numpy()[start:stop],
    'Lead_Velocity': Leader_Velocity,
    'Spacing_VTG': vtg_space_headway_arr,
    'Velocity_VTG': vtg_vel_arr,
    'Acceleration_VTG': vtg_ego_acc_arr,
    'Time_Gap_Comm_VTG': vtg_time_gap_arr,
    'Spacing_SMC': smc_space_headway_arr,
    'Velocity_SMC': smc_vel_arr,
    'Acceleration_SMC': smc_ego_acc_arr,
    'Spacing_ACC': ctg_space_headway_arr,
    'Velocity_ACC': ctg_vel_arr,
    'Acceleration_ACC': ctg_ego_acc_arr,
    'Time_Gap_Comm_ACC': ctg_time_gap_arr,
    'Spacing_Data': (df[['posDiff1', 'posDiff2', 'posDiff3', 'posDiff4']].to_numpy().T)[:, start:stop],
    'Velocity_Data': (df[['Speed2', 'Speed3', 'Speed4', 'Speed5']].to_numpy().T)[:, start:stop],
    'Acceleration_Data': (df[['acceleration2', 'acceleration3', 'acceleration4', 'acceleration5']].to_numpy().T)[:, start:stop-1],
    'Time_Headway_Data': (df[['timeHeadway1', 'timeHeadway2', 'timeHeadway3', 'timeHeadway4']].to_numpy().T)[:, start:stop],
    'Time_Gap_Data': (df[['timeGap1', 'timeGap2', 'timeGap3', 'timeGap4']].to_numpy().T)[:, start:stop-1],
}

savemat('results_for_matlab_plotting.mat', mat_results)