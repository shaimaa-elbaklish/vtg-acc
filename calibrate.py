import numpy as np
import lmfit as lm
import pandas as pd

from acc_model import ACCModelConfig
from calibration import CalibrationConfig, ExperimentConfig

"""
FILES_ACC_MAX = ['ASta_040719_platoon4']
FILES_ACC_MIN = [
    'ASta_050719_platoon1', 'ASta_050719_platoon2', 'ASta_040719_platoon5', 'ASta_040719_platoon6', 
    'ASta_040719_platoon7', 'ASta_040719_platoon8', 'ASta_040719_platoon9'
]
FILES_ACC_HUMAN = ['ASta_040719_platoon3', 'ASta_040719_platoon10']
TRAIN_ACC_IDX, TEST_ACC_IDX = 0, 1
"""
FILES_ZALZONE_ACC_MIN = ['ZalaZONE_dynamic_part1', 'ZalaZONE_dynamic_part4', 'ZalaZONE_dynamic_part5', 'ZalaZONE_dynamic_part6', 
                         'ZalaZONE_dynamic_part7', 'ZalaZONE_dynamic_part10', 'ZalaZONE_dynamic_part12', 'ZalaZONE_dynamic_part13',
                         'ZalaZONE_dynamic_part14', 'ZalaZONE_dynamic_part17', 'ZalaZONE_dynamic_part20', 'ZalaZONE_dynamic_part21',
                         'ZalaZONE_dynamic_part22', 'ZalaZONE_dynamic_part23', 'ZalaZONE_dynamic_part26']
FILES_ZALZONE_ACC_MAX = ['ZalaZONE_dynamic_part2', 'ZalaZONE_dynamic_part3', 'ZalaZONE_dynamic_part11', 'ZalaZONE_dynamic_part15',
                         'ZalaZONE_dynamic_part16', 'ZalaZONE_dynamic_part18', 'ZalaZONE_dynamic_part19', 'ZalaZONE_dynamic_part24',
                         'ZalaZONE_dynamic_part25']
USE_ACC_MIN = True
TRAIN_ACC_IDX, TEST_ACC_IDX = 0, 1
if USE_ACC_MIN:
    filenames = [FILES_ZALZONE_ACC_MIN[TRAIN_ACC_IDX], FILES_ZALZONE_ACC_MIN[TEST_ACC_IDX]]
else:
    filenames = [FILES_ZALZONE_ACC_MAX[TRAIN_ACC_IDX], FILES_ZALZONE_ACC_MAX[TEST_ACC_IDX]]


df_train = pd.read_csv(f'./openacc_data/{filenames[0]}.csv', skiprows=5)
df_train.drop([0, 1, 2], inplace=True)
df_train.interpolate(method='cubic', limit_direction='both', inplace=True)
df_train_note = pd.read_csv(f'./openacc_data/{filenames[0]}.csv', delimiter=';', nrows=5, header=None)
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


df_test = pd.read_csv(f'./openacc_data/{filenames[1]}.csv', skiprows=5)
df_test.drop([0, 1, 2], inplace=True)
df_test.interpolate(method='cubic', limit_direction='both', inplace=True)
df_test_note = pd.read_csv(f'./openacc_data/{filenames[1]}.csv', delimiter=';', nrows=5, header=None)
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
acc_followers = []

params = lm.create_params(
    k0 = {'value': 0.10, 'min': 0.01, 'max': 5},
    k1 = {'value': 0.23, 'min': 0.01, 'max': 5},
    k2 = {'value': 0.07, 'min': 0.01, 'max': 5},
    v0 = {'value': 30, 'min': 30, 'max': 35},
    tg = {'value': 1.2, 'min': 0.1, 'max': 3},
    s0 = {'value': 3.0, 'min': 1, 'max': 5, 'vary': False},
    veh_length = {'value': 5.0, 'vary': False}
)

calib_config = CalibrationConfig(
    parameters = params,
    goodness_of_fit_measure = 'nrmse',
    fit_method = "differential_evolution",
    weight_acc = 1.0,
    weight_vel = 1.0,
    weight_sp = 1.0
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
    
    exp_config = ExperimentConfig(
        model_config = acc_config,
        calibration_config = calib_config,
        train_data = train_data_dict,
        test_data = test_data_dict,
        num_runs = 5
    )
    exp = exp_config.create()


    # run experiment and report best results
    _, best_fit_params = exp.calibrate(verbose=False, validate=True)
    exp.cross_validate(best_fit_params)

    # save calibrated parameters
    exp._model.set_parameters_from_dict(best_fit_params)
    acc_followers.append(exp._model)
    exp._model.save_parameters_to_json(f"./calibration_results/Zalazone_{train_data_dict['veh_type']}_best_fit_parameters.json")
