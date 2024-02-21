import numpy as np
import lmfit as lm

from geneticalgorithm import geneticalgorithm as ga

from dataclasses import dataclass
from typing import Tuple, Optional

from car_following_model import CFModel
from utils_plotting import plot_calibration_validation


class Calibrator:
    def __init__(self, config: 'CalibrationConfig', model: CFModel) -> None:
        self.config = config
        self._model = model
        self._params = config.parameters
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

        return self.config.weight_acc * nrmse_a + self.config.weight_vel * nrmse_v + self.config.weight_sp * nrmse_s

    def _mse(self, v_data: np.ndarray, s_data: np.ndarray, a_data: np.ndarray, 
             v_model: np.ndarray, s_model: np.ndarray, a_model: np.ndarray) -> np.ndarray:
        # exclude the initial points to make vectors the same length
        res_vect = np.vstack((a_data - a_model, v_data[1:] - v_model[1:], s_data[1:] - s_model[1:]))
        return res_vect

    def _objective(self, params: dict, leader_velocity_data: np.ndarray, ego_velocity_data: np.ndarray,
                   space_headway_data: np.ndarray, accel_comm_data: np.ndarray):
        a_model, s_model, v_model = self._model.simulate_trajectory_from_params_dict(
            params, leader_velocity_data, ego_velocity_data[0], space_headway_data[0], 
            dt=self.config.integration_step, integration_method=self.config.integration_method
        )
        if self._gof_measure == "mse":
            return self._residuals_mse(ego_velocity_data, space_headway_data, accel_comm_data, v_model, s_model, a_model)
        elif self._gof_measure == "nrmse":
            return self._nrmse(ego_velocity_data, space_headway_data, accel_comm_data, v_model, s_model, a_model)
        else:
            raise NotImplementedError

    def fit(self, leader_velocity_data: np.ndarray, ego_velocity_data: np.ndarray,
            space_headway_data: np.ndarray, accel_comm_data: np.ndarray,
            time: Optional[np.ndarray] = None, verbose: bool = False, validate: bool = True) -> Tuple[float, dict]:
        out = lm.minimize(
            fcn=self._objective, params=self._params, method=self._fit_method, 
            args=(leader_velocity_data, ego_velocity_data, space_headway_data, accel_comm_data)
        )
        if verbose:
            print(lm.fit_report(out))
            print(lm.fit_report(out.params))
            print(out.params.pretty_print())

        if validate:
            a_model, s_model, v_model = self._model.simulate_trajectory_from_params_dict(
                out.params, leader_velocity_data, ego_velocity_data[0], space_headway_data[0], 
                dt=self.config.integration_step, integration_method=self.config.integration_method
            )
            # gof_val = self._nrmse(ego_velocity_trajectory, space_headway_trajectory, acceleration_command, v_model, s_model, a_model)
            print(f'Goodness of fit value = {np.mean(out.residual):.4f}')
            if time is None:
                time = np.arange(start=0, stop=leader_velocity_data.shape[0])
            plot_calibration_validation(
                time, leader_velocity_data, ego_velocity_data, space_headway_data, accel_comm_data, v_model, s_model, a_model
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
    integration_step: float = 0.1
    integration_method: str = 'euler'

    def create(self, model: CFModel) -> Calibrator:
        return Calibrator(self, model)
    


class GACalibrator:
    def __init__(self, config: 'GACalibrationConfig', model: CFModel) -> None:
        self.config = config
        self._model = model
        self._params = config.parameters
        self._param_keys = list(self._params.keys())
        self._gof_measure = config.goodness_of_fit_measure
        self.data_trajectories = {
            'lead_vel': None,
            'ego_vel': None,
            'space_headway': None,
            'ego_acc': None
        }
        self.algorithm_param = {
            'max_num_iteration': 500,
            'population_size': 100,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type':'uniform',
            'max_iteration_without_improv': 100
        }
    
    def _array_to_dict(self, X: np.ndarray) -> dict:
        curr_dict = {}
        for i in range(X.shape[0]):
            curr_dict[self._param_keys[i]] = X[i]
        return curr_dict

    def _objective(self, X: np.ndarray) -> float:
        curr_params = self._array_to_dict(X)
        a_model, s_model, v_model = self._model.simulate_trajectory_from_params_dict(
            curr_params, self.data_trajectories['lead_vel'], 
            self.data_trajectories['ego_vel'][0], self.data_trajectories['space_headway'][0], 
            dt=self.config.integration_step, integration_method=self.config.integration_method
        )
        if self._gof_measure == "mse":
            pass
        elif self._gof_measure == "nrmse":
            rmse_a = np.sqrt(np.sum(np.power(self.data_trajectories['ego_acc'] - a_model, 2)) / self.data_trajectories['ego_acc'].shape[0])
            rmse_v = np.sqrt(np.sum(np.power(self.data_trajectories['ego_vel'] - v_model, 2)) / self.data_trajectories['ego_vel'].shape[0])
            rmse_s = np.sqrt(np.sum(np.power(self.data_trajectories['space_headway'] - s_model, 2)) / self.data_trajectories['space_headway'].shape[0])

            nrmse_a = rmse_a / np.sqrt(np.sum(np.power(self.data_trajectories['ego_acc'], 2)) / self.data_trajectories['ego_acc'].shape[0])
            nrmse_v = rmse_v / np.sqrt(np.sum(np.power(self.data_trajectories['ego_vel'], 2)) / self.data_trajectories['ego_vel'].shape[0])
            nrmse_s = rmse_s / np.sqrt(np.sum(np.power(self.data_trajectories['space_headway'], 2)) / self.data_trajectories['space_headway'].shape[0])

            return self.config.weight_acc * nrmse_a + self.config.weight_vel * nrmse_v + self.config.weight_sp * nrmse_s
        else:
            raise NotImplementedError


    def fit(self, leader_velocity_data: np.ndarray, ego_velocity_data: np.ndarray,
            space_headway_data: np.ndarray, accel_comm_data: np.ndarray,
            time: Optional[np.ndarray] = None, 
            verbose: bool = False, validate: bool = True) -> Tuple[float, dict]:
        self.data_trajectories['lead_vel'] = leader_velocity_data
        self.data_trajectories['ego_vel'] = ego_velocity_data
        self.data_trajectories['space_headway'] = space_headway_data
        self.data_trajectories['ego_acc'] = accel_comm_data

        n_dim = len(self._params)
        var_bounds = np.zeros(shape=(n_dim, 2))
        for i in range(n_dim):
            var_bounds[i, 0] = self._params[self._param_keys[i]]['min']
            var_bounds[i, 1] = self._params[self._param_keys[i]]['max']
        
        ga_model = ga(
            function=self._objective, dimension=n_dim, variable_type='real', variable_boundaries=var_bounds,
            algorithm_parameters=self.algorithm_param, function_timeout=30
        )
        
        results_dict = {'best_fn': [], 'best_vars': []}
        for r in range(self.config.num_runs):
            ga_model.run()
            results_dict['best_fn'].append(ga_model.best_function)
            results_dict['best_vars'].append(ga_model.best_variable)
        best_idx = np.argmin(results_dict['best_fn'])

        out_residuals = results_dict['best_fn'][best_idx]
        if verbose:
            print(results_dict['best_fn'])
        out_params = self._array_to_dict(results_dict['best_vars'][best_idx])
        if validate:
            a_model, s_model, v_model = self._model.simulate_trajectory_from_params_dict(
                out_params, leader_velocity_data, ego_velocity_data[0], space_headway_data[0], 
                dt=self.config.integration_step, integration_method=self.config.integration_method
            )
            print(f'Goodness of fit value = {out_residuals:.4f}')
            if time is None:
                time = np.arange(start=0, stop=leader_velocity_data.shape[0])
            plot_calibration_validation(
                time, leader_velocity_data, ego_velocity_data, space_headway_data, accel_comm_data, v_model, s_model, a_model
            )

        return out_residuals, out_params


@dataclass
class GACalibrationConfig:
    parameters: dict # Parameters of the CF model to be calibrated
    goodness_of_fit_measure: str = "nrmse" # Goodness of fit measure, options: ["mse", "nrmse"]
    weight_acc: float = 1.0 # weights for NRMSE of acceleration trajectory
    weight_vel: float = 1.0 # weights for NRMSE of velocity trajectory
    weight_sp: float = 1.0 # weights for NRMSE of space headway trajectory
    integration_step: float = 0.1
    integration_method: str = 'euler'
    num_runs: int = 5

    def create(self, model: CFModel) -> GACalibrator:
        return GACalibrator(self, model)