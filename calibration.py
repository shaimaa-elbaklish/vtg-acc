import numpy as np
import lmfit as lm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from typing import Tuple, Optional

from model import CFModel, CFModelConfig
from utils_plotting import plot_calibration_validation

FILES_ACC_MAX = ['ASta_040719_platoon4']
FILES_ACC_MIN = [
    'ASta_050719_platoon1', 'ASta_050719_platoon2', 'ASta_040719_platoon5', 'ASta_040719_platoon6', 
    'ASta_040719_platoon7', 'ASta_040719_platoon8', 'ASta_040719_platoon9'
]
FILES_ACC_HUMAN = ['ASta_040719_platoon3', 'ASta_040719_platoon10']


class Calibrator:
    def __init__(self, config: "CalibrationConfig", model: CFModel) -> None:
        self._config = config
        self._model = model
        self._parameters = config.parameters
        self._gof_measure = config.goodness_of_fit_measure
        self._fit_method = config.fit_method
        if self._fit_method == "leastsq":
            try:
                assert(self._gof_measure == "mse")
            except:
                print("To use least squares fitting, the GOF measure should return numpy.ndarray of residuals." +
                      "\nChanged GOF measure to MSE.")
                self._gof_measure = "mse"

    def _nrmse(self, v_data: np.ndarray, s_data: np.ndarray, a_data: np.ndarray, 
               v_model: np.ndarray, s_model: np.ndarray, a_model: np.ndarray) ->  float:
        assert(v_data.shape[0] == v_model.shape[0])
        assert(s_data.shape[0] == s_model.shape[0])
        assert(a_data.shape[0] == a_model.shape[0])
        rmse_a = np.sqrt(np.sum(np.power(a_data - a_model, 2)) / a_data.shape[0])
        rmse_v = np.sqrt(np.sum(np.power(v_data - v_model, 2)) / v_data.shape[0])
        rmse_s = np.sqrt(np.sum(np.power(s_data - s_model, 2)) / s_data.shape[0])

        nrmse_a = rmse_a / np.sqrt(np.sum(np.power(a_data, 2)) / a_data.shape[0])
        nrmse_v = rmse_v / np.sqrt(np.sum(np.power(v_data, 2)) / v_data.shape[0])
        nrmse_s = rmse_s / np.sqrt(np.sum(np.power(s_data, 2)) / s_data.shape[0])

        return self._config.weight_acc * nrmse_a + self._config.weight_vel * nrmse_v + self._config.weight_sp * nrmse_s


    def _residuals_mse(self, v_data: np.ndarray, s_data: np.ndarray, a_data: np.ndarray, 
                       v_model: np.ndarray, s_model: np.ndarray, a_model: np.ndarray) -> np.ndarray:
        # exclude the initial points to make vectors the same length
        res_vect = np.vstack((a_data - a_model, v_data[1:] - v_model[1:], s_data[1:] - s_model[1:]))
        return res_vect
    
    def _generate_trajectory_from_dict(self, params: dict, leader_velocity_trajectory: np.ndarray, ego_velocity_trajectory: np.ndarray,
                                       space_headway_trajectory: np.ndarray, validate: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = leader_velocity_trajectory.shape[0]
        v_model = np.zeros(shape=(N,))
        s_model = np.zeros(shape=(N,))
        a_model = np.zeros(shape=(N-1,))
        v_model[0], s_model[0] = ego_velocity_trajectory[0], space_headway_trajectory[0]
        for t in range(N-1):
            a_model[t], s_model[t+1], v_model[t+1] = self._model.step_using_parameters_from_dict(
                params, leader_velocity_trajectory[t], ego_velocity_trajectory[t], space_headway_trajectory[t], validate
            )
        return a_model, s_model, v_model
    
    def _objective(self, params: dict, leader_velocity_trajectory: np.ndarray, ego_velocity_trajectory: np.ndarray,
                   space_headway_trajectory: np.ndarray, acceleration_command: np.ndarray):
        a_model, s_model, v_model = self._generate_trajectory_from_dict(params, leader_velocity_trajectory,
                                                                        ego_velocity_trajectory, space_headway_trajectory, validate=False)
        if self._gof_measure == "mse":
            return self._residuals_mse(ego_velocity_trajectory, space_headway_trajectory, acceleration_command, v_model, s_model, a_model)
        elif self._gof_measure == "nrmse":
            return self._nrmse(ego_velocity_trajectory, space_headway_trajectory, acceleration_command, v_model, s_model, a_model)
        else:
            raise NotImplementedError

    def fit(self, leader_velocity_trajectory: np.ndarray, ego_velocity_trajectory: np.ndarray,
            space_headway_trajectory: np.ndarray, acceleration_command: np.ndarray,
            time: Optional[np.ndarray] = None, verbose: bool = False, validate: bool = True) -> dict:
        """
        Fits the CF model to the real data supplied and returns the best-fit parameters
        Parameters:
        -----------
            leader_velocity_trajectory: numpy.ndarray, shape=(N,)
                Trajectory of leader vehcile velocity from data
            ego_velocity_trajectory: numpy.ndarray, shape=(N,)
                Trajectory of ego vehicle velocity from data
            space_headway_trajectory: numpy.ndarray, shape=(N,)
                Trajectory of space headway between ego and leader vehicles from data
            acceleration_command: numpy.ndarray, shape=(N-1,)
                Trajectory of ego vehicle acceleration from data
            time: numpy.ndarray, shape=(N,), Optional
            verbose: bool, default = False
                flag to print reports of lmfit.minimize results
            validate: bool, default = True
                flag to validate results (NRMSE and plotting)
        Returns:
        --------
            residuals: numpy.ndarray
            best_fit_parameters: dict
        """
        out = lm.minimize(
            self._objective, self._parameters, method=self._fit_method, 
            args=(leader_velocity_trajectory, ego_velocity_trajectory, space_headway_trajectory, acceleration_command)
        )
        if verbose:
            print(lm.fit_report(out))
            print(lm.fit_report(out.params))
            print(out.params.pretty_print())
        if validate:
            a_model, s_model, v_model = self._generate_trajectory_from_dict(
                out.params, leader_velocity_trajectory, ego_velocity_trajectory, space_headway_trajectory, validate=True)
            # gof_val = self._nrmse(ego_velocity_trajectory, space_headway_trajectory, acceleration_command, v_model, s_model, a_model)
            print(f'Goodness of fit value = {np.mean(out.residual):.4f}')
            if time is None:
                time = np.arange(start=0, stop=leader_velocity_trajectory.shape[0])
            plot_calibration_validation(
                time, leader_velocity_trajectory, ego_velocity_trajectory, space_headway_trajectory, acceleration_command, v_model, s_model, a_model
            )
        return out.residual, out.params


@dataclass
class CalibrationConfig:
    """
        Configuration parameters for caliberation of ACC model
    """
    parameters: dict # Parameters of the CF model to be calibrated
    goodness_of_fit_measure: str = "nrmse" # Goodness of fit measure, options: ["mse", "nrmse"]
    fit_method: str = "differential_evolution" # Optimization method to be used in lmfit.minimize
    weight_acc: float = 1.0 # weights for NRMSE of acceleration trajectory
    weight_vel: float = 1.0 # weights for NRMSE of velocity trajectory
    weight_sp: float = 1.0 # weights for NRMSE of space headway trajectory

    def create(self, model: CFModel) -> Calibrator:
        return Calibrator(self, model)


class CalibrationExperiment:
    def __init__(self, config: "ExperimentConfig", model: CFModel, calibrator: Calibrator) -> None:
        self._model = model
        self._calib = calibrator
        self._train_data = config.train_data
        self._test_data = config.test_data
        self._num_runs = config.num_runs
        self._results = { 'gof': [], 'params': [] }
    
    def calibrate(self, verbose: bool = False, validate: bool = True) -> Tuple[float, dict]:
        """
        Runs the calibration experiment multiple times and report optimal results.
        Also reports the coefficient of variation of the GOF across runs.
        """
        for e in range(self._num_runs):
            print(f'\n \n Calibration Experiment {e+1}:')
            out_residuals, out_params = self._calib.fit(
                self._train_data['lead_vel'],
                self._train_data['ego_vel'],
                self._train_data['space_headway'],
                self._train_data['ego_acc'],
                self._train_data['time'],
                verbose, validate=False
            )
            self._results['gof'].append(np.mean(out_residuals))
            self._results['params'].append(out_params)
            print(f"Achieved GOF value = {self._results['gof'][e]:.4f}.")
        
        idx = self._results['gof'].index(min(self._results['gof']))
        cv = np.std(self._results['gof']) / np.mean(self._results['gof'])
        print(f"The best-fit parameters present a GOF value = {self._results['gof'][idx]:.4f}")
        print(f'The Coefficient of Variation across {self._num_runs} experiments = {cv:.4f}.')
        if validate:
            a_model, s_model, v_model = self._calib._generate_trajectory_from_dict(
                self._results['params'][idx], 
                self._train_data['lead_vel'], self._train_data['ego_vel'], self._train_data['space_headway'], 
                validate=True
            )
            plot_calibration_validation(
                self._train_data['time'], self._train_data['lead_vel'], 
                self._train_data['ego_vel'], self._train_data['space_headway'], self._train_data['ego_acc'], 
                v_model, s_model, a_model,
                fig_title="Validation Results"
            )
        return self._results['gof'][idx], self._results['params'][idx]

    def cross_validate(self, best_fit_params: Optional[dict] = None) -> Optional[dict]:
        print('\n \n Cross-Validation:')
        if best_fit_params is None:
            print('No best-fit parameters supplied. \n Choosing best-fit parameters according to cross-validation score!')
            self._results['gof_cross_val'] = []
            for e in range(self._num_runs):
                a_model, s_model, v_model = self._calib._generate_trajectory_from_dict(
                    self._results['params'][e], 
                    self._test_data['lead_vel'], self._test_data['ego_vel'], self._test_data['space_headway'], 
                    validate=True
                )
                gof_val = self._calib._nrmse(
                    self._test_data['ego_vel'], self._test_data['space_headway'], self._test_data['ego_acc'], 
                    v_model, s_model, a_model
                )
                self._results['gof_cross_val'].append(gof_val)
            idx = self._results['gof_cross_val'].index(min(self._results['gof_cross_val']))
            print(f"Best cross-validation score: GOF value = {self._results['gof_cross_val'][idx]:.4f}.")
            cv = np.std(self._results['gof_cross_val']) / np.mean(self._results['gof_cross_val'])
            print(f'The Coefficient of Variation across {self._num_runs} experiments = {cv:.4f}.')
            a_model, s_model, v_model = self._calib._generate_trajectory_from_dict(
                self._results['params'][idx],
                self._test_data['lead_vel'], self._test_data['ego_vel'], self._test_data['space_headway'], 
                validate=True
            )
            plot_calibration_validation(
                self._test_data['time'], self._test_data['lead_vel'], 
                self._test_data['ego_vel'], self._test_data['space_headway'], self._test_data['ego_acc'], 
                v_model, s_model, a_model,
                fig_title="Cross-Validation Results"
            )
            return self._results['params'][idx]
        else:
            a_model, s_model, v_model = self._calib._generate_trajectory_from_dict(
                best_fit_params, 
                self._test_data['lead_vel'], self._test_data['ego_vel'], self._test_data['space_headway'], 
                validate=True
            )
            gof_val = self._calib._nrmse(
                self._test_data['ego_vel'], self._test_data['space_headway'], self._test_data['ego_acc'], 
                v_model, s_model, a_model
            )
            print(f'GOF value = {gof_val:.4f}')
            plot_calibration_validation(
                self._test_data['time'], self._test_data['lead_vel'], 
                self._test_data['ego_vel'], self._test_data['space_headway'], self._test_data['ego_acc'], 
                v_model, s_model, a_model,
                fig_title="Cross-Validation Results"
            )
            return None


@dataclass(frozen=True)
class ExperimentConfig:
    model_config: CFModelConfig
    calibration_config: CalibrationConfig
    train_data: dict[str, np.ndarray]
    test_data: dict[str, np.ndarray]
    num_runs: int = 10

    def create(self) -> CalibrationExperiment:
        model = self.model_config.create()
        calibrator = self.calibration_config.create(model)
        return CalibrationExperiment(self, model, calibrator)
