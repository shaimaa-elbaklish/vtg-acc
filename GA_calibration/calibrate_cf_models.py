import numpy as np
import pandas as pd
import lmfit as lm

from calibrator import CalibrationConfig
from car_following_model import IDMConfig


FILES_ACC_MAX = ['ASta_040719_platoon4']
FILES_ACC_MIN = [
    'ASta_050719_platoon1', 'ASta_050719_platoon2', 'ASta_040719_platoon5', 'ASta_040719_platoon6', 
    'ASta_040719_platoon7', 'ASta_040719_platoon8', 'ASta_040719_platoon9'
]
FILES_ACC_HUMAN = ['ASta_040719_platoon3', 'ASta_040719_platoon10']
TRAIN_ACC_IDX, TEST_ACC_IDX = 0, 1
filenames = [FILES_ACC_MIN[TRAIN_ACC_IDX], FILES_ACC_MIN[TEST_ACC_IDX]]

df_train = pd.read_csv(f'./data/{filenames[0]}.csv', skiprows=5)
if df_train.isnull().any().any():
    df_train.drop([0, 1, 2], inplace=True)
    df_train.interpolate(method='cubic', limit_direction='both', inplace=True)
    df_train.reset_index(inplace=True)
df_train_note = pd.read_csv(f'./data/{filenames[0]}.csv', delimiter=';', nrows=5, header=None)
num_veh = int(df_train_note.loc[2,0].split(',')[1])
train_veh_types = df_train_note.loc[1,0].split(',')
train_veh_types = [x for x in train_veh_types if x != '']
train_veh_types = train_veh_types[1:]
freq = 10
for c in range(1, num_veh):
    df_train[f"posDiff{c}"] = np.sqrt((df_train[f"E{c}"] - df_train[f"E{c+1}"])**2 + (df_train[f"N{c}"] - df_train[f"N{c+1}"])**2) #+ (df_train[f"U{c}"] - df_train[f"U{c+1}"])**2)
    df_train[f"acceleration{c+1}"] = df_train[f"Speed{c+1}"].diff().shift(-1).fillna(0) * freq
# df_train["Time"] = df_train["Time"] - df_train.loc[0, "Time"]
train_data_dict = {
    'veh_type': "",
    'time': df_train['Time'].to_numpy(),
    'lead_vel': np.empty(shape=(len(df_train),)),
    'ego_vel': np.empty(shape=(len(df_train),)),
    'space_headway': np.empty(shape=(len(df_train),)),
    'ego_acc': np.empty(shape=(len(df_train)-1,)),
}


df_test = pd.read_csv(f'./data/{filenames[1]}.csv', skiprows=5)
if df_test.isnull().any().any():
    df_test.drop([0, 1, 2], inplace=True)
    df_test.interpolate(method='cubic', limit_direction='both', inplace=True)
    df_test.reset_index(inplace=True)
df_test_note = pd.read_csv(f'./data/{filenames[1]}.csv', delimiter=';', nrows=5, header=None)
test_veh_types = df_test_note.loc[1,0].split(',')
test_veh_types = [x for x in test_veh_types if x != '']
test_veh_types = test_veh_types[1:]
num_veh_test = int(df_test_note.loc[2,0].split(',')[1])
for c in range(1, num_veh_test):
    df_test[f"posDiff{c}"] = np.sqrt((df_test[f"E{c}"] - df_test[f"E{c+1}"])**2 + (df_test[f"N{c}"] - df_test[f"N{c+1}"])**2) #+ (df_test[f"U{c}"] - df_test[f"U{c+1}"])**2)
    if df_test[f"posDiff{c}"].isnull().any():
        print(c, df_test[f"posDiff{c}"][df_test[f"posDiff{c}"].isnull()])
        exit(1)
    df_test[f"acceleration{c+1}"] = df_test[f"Speed{c+1}"].diff().shift(-1).fillna(0) * freq
test_data_dict = {
    'veh_type': "",
    'time': df_test['Time'].to_numpy(),
    'lead_vel': np.empty(shape=(len(df_test),)),
    'ego_vel': np.empty(shape=(len(df_test),)),
    'space_headway': np.empty(shape=(len(df_test),)),
    'ego_acc': np.empty(shape=(len(df_test)-1,)),
}


"""
Prepare Car-Following Model
"""
model_config = IDMConfig(
    desired_speed = 30,
    min_gap = 2.0,
    time_headway = 1.5,
    max_acceleration_idm = 0.3,
    comfortable_deceleration = 3.0,
    velocity_exponent = 4.0,
    use_acceleration_constraints = True,
    max_acceleration = 5.0,
    min_acceleration = -5.0
)

params = lm.create_params(
    #k0 = {'value': 0.10, 'min': 0.01, 'max': 5},
    #k1 = {'value': 0.23, 'min': 0.01, 'max': 5},
    #k2 = {'value': 0.07, 'min': 0.01, 'max': 5},
    desired_speed = {'value': 30, 'min': 30, 'max': 35},
    #time_gap = {'value': 1.2, 'min': 0.1, 'max': 3},
    # min_gap = {'value': 3.0, 'min': 1, 'max': 5, 'vary': False},
    time_headway  = {'value': 1.5, 'min': 0.1, 'max': 3},
    delta  = {'value': 4, 'min': 0.1, 'max': 10, 'vary': False},
    max_accel_idm = {'value': 0.3, 'min': 0.1, 'max': 5},
    comfort_decel = {'value': 3, 'min': 0.1, 'max': 5}
)

calib_config = CalibrationConfig(
    parameters = params,
    goodness_of_fit_measure = "nrmse",
    fit_method = "differential_evolution",
    weight_acc = 1.0,
    weight_vel = 1.0,
    weight_sp = 1.0,
    integration_step = 0.1,
    integration_method = 'euler'
)


for c in range(num_veh-1):

    train_data_dict['veh_type'] = train_veh_types[c+1]
    train_data_dict['lead_vel'] = df_train[f'Speed{c+1}'].to_numpy()
    train_data_dict['ego_vel'] = df_train[f'Speed{c+2}'].to_numpy()
    train_data_dict['ego_acc'] = df_train[f'acceleration{c+2}'].to_numpy()[:-1]
    train_data_dict['space_headway'] = df_train[f'posDiff{c+1}'].to_numpy()

    test_idx = test_veh_types.index(train_data_dict['veh_type'])
    test_data_dict['veh_type'] = test_veh_types[test_idx]
    assert(test_data_dict['veh_type'] == train_data_dict['veh_type'])
    test_data_dict['lead_vel'] = df_test[f'Speed{test_idx}'].to_numpy()
    test_data_dict['ego_vel'] = df_test[f'Speed{test_idx+1}'].to_numpy()
    test_data_dict['ego_acc'] = df_test[f'acceleration{test_idx+1}'].to_numpy()[:-1]
    test_data_dict['space_headway'] = df_test[f'posDiff{test_idx}'].to_numpy()

    cf_model = model_config.create()
    calib = calib_config.create(cf_model)

    out_res, out_params = calib.fit(
        train_data_dict['lead_vel'], train_data_dict['ego_vel'], train_data_dict['space_headway'], train_data_dict['ego_acc'], 
        time=train_data_dict['time'], verbose=True, validate=False
    )

    cf_model.set_params_from_dict(out_params)
    cf_model.save_params_to_json(f"./calibration_results/AstaZero_{train_data_dict['veh_type']}_IDM_best_fit.json", np.mean(out_res))
    


