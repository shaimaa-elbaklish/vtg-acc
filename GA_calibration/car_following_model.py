import json
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from dataclasses import dataclass


class CFModel(ABC):
    def __init__(self):
        super(CFModel, self).__init__()
    
    @abstractmethod
    def get_acceleration_command(self, leader_velocity: float, 
                                 ego_velocity: float, space_headway: float) -> float:
        pass

    @abstractmethod
    def step(self, leader_velocity: float, ego_velocity: float, space_headway: float, acc_command: Optional[float] = None,
             dt: float = 0.1, integration_method: str = "euler") -> Tuple[float, float]:
        pass
    
    @abstractmethod
    def simulate_trajectory(self, leader_velocity_traj: np.ndarray, 
                            init_ego_velocity: float, init_space_headway: float,
                            dt: float = 0.1, time: Optional[np.ndarray] = None, integration_method: str = "euler",
                            plot_trajectory: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def set_params_from_dict(self, params:dict) -> None:
        pass

    @abstractmethod
    def simulate_trajectory_from_params_dict(self, params: dict, leader_velocity_traj: np.ndarray, 
                                        init_ego_velocity: float, init_space_headway: float,
                                        dt: float = 0.1, integration_method: str = "euler") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


@dataclass
class CFModelConfig(ABC):

    @abstractmethod
    def create(self) -> CFModel:
        pass

#######################################################################################
#######################################################################################
"""
Intelligent Driver Model
"""
class IDM(CFModel):
    def __init__(self, config: 'IDMConfig'):
        super(IDM, self).__init__()
        self._a = config.max_acceleration_idm
        self._b = config.comfortable_deceleration
        self._th = config.time_headway
        self._s0 = config.min_gap
        self._v0 = config.desired_speed
        self._delta = config.velocity_exponent
        self.config = config
    
    def get_acceleration_command(self, leader_velocity: float, 
                                 ego_velocity: float, space_headway: float) -> float:
        d_vel = leader_velocity - ego_velocity
        s_des = self._s0 + max(0, self._th*ego_velocity - ego_velocity*d_vel/(2*np.sqrt(self._a*self._b)))
        acc_comm = self._a*(1 - np.power(ego_velocity/self._v0, self._delta) - np.square(s_des/space_headway))
        if self.config.use_acceleration_constraints:
            acc_comm = max(min(acc_comm, self.config.max_acceleration), self.config.min_acceleration)
        return acc_comm
    
    def step(self, leader_velocity: float, ego_velocity: float, space_headway: float, acc_command: Optional[float] = None,
             dt: float = 0.1, integration_method: str = "euler") -> Tuple[float, float]:
        if acc_command is None:
            acc_command = self.get_acceleration_command(leader_velocity, ego_velocity, space_headway)
        if integration_method == 'euler':
            s_next = space_headway + dt*(leader_velocity - ego_velocity)
            v_next = ego_velocity + dt*acc_command
            return s_next, v_next
        else:
            raise NotImplementedError
    
    def simulate_trajectory(self, leader_velocity_traj: np.ndarray, 
                            init_ego_velocity: float, init_space_headway: float,
                            dt: float = 0.1, time: Optional[np.ndarray] = None, integration_method: str = "euler",
                            plot_trajectory: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_steps = leader_velocity_traj.shape[0]
        ego_velocity_traj = np.zeros(shape=(num_steps,))
        space_headway_traj = np.zeros(shape=(num_steps,))
        ego_accel_traj = np.zeros(shape=(num_steps-1,))

        ego_velocity_traj[0] = init_ego_velocity
        space_headway_traj[0] = init_space_headway

        for i in range(num_steps-1):
            ego_accel_traj[i] = self.get_acceleration_command(leader_velocity_traj[i], ego_velocity_traj[i], space_headway_traj[i])
            space_headway_traj[i+1], ego_velocity_traj[i+1] = self.step(leader_velocity_traj[i], 
                                                                        ego_velocity_traj[i], space_headway_traj[i], ego_accel_traj[i], 
                                                                        dt=dt, integration_method=integration_method)
        
        if plot_trajectory:
            raise NotImplementedError
        
        return ego_accel_traj, space_headway_traj, ego_velocity_traj
    
    def set_params_from_dict(self, params:dict, is_ga: bool = False) -> None:
        if is_ga:
            try:
                self._v0 = params['desired_speed']
                #self._s0 = params['min_gap']
                self._th = params['time_headway']
                self._a = params['max_accel_idm']
                self._b = params['comfort_decel']
                self._delta = params['delta']
            except:
                print('ERROR: Key not found in parameters dictionary!')
            return
        try:
            self._v0 = params['desired_speed'].value
            #self._s0 = params['min_gap'].value
            self._th = params['time_headway'].value
            self._a = params['max_accel_idm'].value
            self._b = params['comfort_decel'].value
            self._delta = params['delta'].value
        except:
            print('ERROR: Key not found in parameters dictionary!')
    
    def save_params_to_json(self, file_path: str, calibration_gof: Optional[float] = None) -> None:
        params = {
            'max_accel_idm': self._a,
            'comfort_decel': self._b,
            'delta': self._delta,
            'min_gap': self._s0,
            'time_headway': self._th,
            'desired_speed': self._v0,
            'use_acceleration_constraints': self.config.use_acceleration_constraints
        }
        if self.config.use_acceleration_constraints:
            params['min_acceleration'] = self.config.min_acceleration
            params['max_acceleration'] = self.config.max_acceleration
        if calibration_gof is not None:
            params['calib_gof_value'] = calibration_gof
        with open(file_path, 'w') as outfile:
            json.dump(params, outfile)
    
    def simulate_trajectory_from_params_dict(self, params: dict, leader_velocity_traj: np.ndarray, 
                                        init_ego_velocity: float, init_space_headway: float,
                                        dt: float = 0.1, integration_method: str = "euler") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _v0 = params['desired_speed']
        #_s0 = params['min_gap']
        _th = params['time_headway']
        _a = params['max_accel_idm']
        _b = params['comfort_decel']
        _delta = params['delta']
        
        num_steps = leader_velocity_traj.shape[0]
        ego_velocity_traj = np.zeros(shape=(num_steps,))
        space_headway_traj = np.zeros(shape=(num_steps,))
        ego_accel_traj = np.zeros(shape=(num_steps-1,))

        ego_velocity_traj[0] = init_ego_velocity
        space_headway_traj[0] = init_space_headway

        for i in range(num_steps-1):
            d_vel = leader_velocity_traj[i] - ego_velocity_traj[i]
            s_des = self._s0 + max(0, _th*ego_velocity_traj[i] - ego_velocity_traj[i]*d_vel/(2*np.sqrt(_a*_b)))
            v_ratio = np.float_power(np.abs(ego_velocity_traj[i])/_v0, _delta)
            ego_accel_traj[i] = _a*(1 - v_ratio - np.square(s_des/space_headway_traj[i]))
            if self.config.use_acceleration_constraints:
                ego_accel_traj[i] = max(min(ego_accel_traj[i], self.config.max_acceleration), self.config.min_acceleration)
            
            if integration_method == 'euler':
                space_headway_traj[i+1] = space_headway_traj[i] + dt*(leader_velocity_traj[i] - ego_velocity_traj[i])
                ego_velocity_traj[i+1] = ego_velocity_traj[i] + dt*ego_accel_traj[i]
            else:
                raise NotImplementedError
        
        return ego_accel_traj, space_headway_traj, ego_velocity_traj


@dataclass
class IDMConfig(CFModelConfig):

    desired_speed: float = 30
    min_gap: float = 2.0
    time_headway: float = 1.5
    max_acceleration_idm: float = 0.3
    comfortable_deceleration: float = 3.0
    velocity_exponent: float = 4.0
    use_acceleration_constraints: bool = True
    max_acceleration: float = 5.0
    min_acceleration: float = -5.0

    def create(self) -> IDM:
        return IDM(self)


#######################################################################################
#######################################################################################
"""
CTG ACC Model
"""
class ACCModel(CFModel):
    def __init__(self, config: 'ACCModelConfig'):
        super(ACCModel, self).__init__()
        self._v0 = config.desired_speed
        self._k0 = config.k0
        self._k1 = config.k1
        self._k2 = config.k2
        self._tg = config.time_gap
        self._s0 = config.min_gap
        self._L = config.vehicle_length
        self.config = config

    def get_acceleration_command(self, leader_velocity: float, 
                                 ego_velocity: float, space_headway: float) -> float:
        s_des = self._s0 + self._L + self._tg*ego_velocity
        acc_comm = min(
            self._k0*(self._v0 - ego_velocity),
            self._k1*(space_headway - s_des) + self._k2*(leader_velocity - ego_velocity)
        )
        if self.config.use_acceleration_constraints:
            acc_comm = max(min(acc_comm, self.config.max_acceleration), self.config.min_acceleration)
        return acc_comm

    def step(self, leader_velocity: float, ego_velocity: float, space_headway: float, acc_command: Optional[float] = None,
             dt: float = 0.1, integration_method: str = "euler") -> Tuple[float, float]:
        if acc_command is None:
            acc_command = self.get_acceleration_command(leader_velocity, ego_velocity, space_headway)
        if integration_method == 'euler':
            s_next = space_headway + dt*(leader_velocity - ego_velocity)
            v_next = ego_velocity + dt*acc_command
            return s_next, v_next
        else:
            raise NotImplementedError
    
    def simulate_trajectory(self, leader_velocity_traj: np.ndarray, 
                            init_ego_velocity: float, init_space_headway: float,
                            dt: float = 0.1, time: Optional[np.ndarray] = None, integration_method: str = "euler",
                            plot_trajectory: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_steps = leader_velocity_traj.shape[0]
        ego_velocity_traj = np.zeros(shape=(num_steps,))
        space_headway_traj = np.zeros(shape=(num_steps,))
        ego_accel_traj = np.zeros(shape=(num_steps-1,))

        ego_velocity_traj[0] = init_ego_velocity
        space_headway_traj[0] = init_space_headway

        for i in range(num_steps-1):
            ego_accel_traj[i] = self.get_acceleration_command(leader_velocity_traj[i], ego_velocity_traj[i], space_headway_traj[i])
            space_headway_traj[i+1], ego_velocity_traj[i+1] = self.step(leader_velocity_traj[i], 
                                                                        ego_velocity_traj[i], space_headway_traj[i], ego_accel_traj[i], 
                                                                        dt=dt, integration_method=integration_method)
        
        if plot_trajectory:
            raise NotImplementedError
        
        return ego_accel_traj, space_headway_traj, ego_velocity_traj

    def set_params_from_dict(self, params:dict, is_ga: bool = False) -> None:
        if is_ga:
            try:
                self._v0 = params['desired_speed']
                # self._s0 = params['min_gap']
                self._tg = params['time_gap']
                self._k0 = params['k0']
                self._k1 = params['k1']
                self._k2 = params['k2']
            except:
                print('ERROR: Key not found in parameters dictionary!')
            return
        try:
            self._v0 = params['desired_speed'].value
            # self._s0 = params['min_gap'].value
            self._tg = params['time_gap'].value
            self._k0 = params['k0'].value
            self._k1 = params['k1'].value
            self._k2 = params['k2'].value
        except:
            print('ERROR: Key not found in parameters dictionary!')

    def save_params_to_json(self, file_path: str, calibration_gof: Optional[float] = None):
        params = {
            'k0': self._k0,
            'k1': self._k1,
            'k2': self._k2,
            'min_gap': self._s0,
            'vehicle_length': self._L,
            'time_gap': self._tg,
            'desired_speed': self._v0,
            'use_acceleration_constraints': self.config.use_acceleration_constraints
        }
        if self.config.use_acceleration_constraints:
            params['min_acceleration'] = self.config.min_acceleration
            params['max_acceleration'] = self.config.max_acceleration
        if calibration_gof is not None:
            params['calib_gof_value'] = calibration_gof
        with open(file_path, 'w') as outfile:
            json.dump(params, outfile)

    def simulate_trajectory_from_params_dict(self, params: dict, leader_velocity_traj: np.ndarray, 
                                        init_ego_velocity: float, init_space_headway: float,
                                        dt: float = 0.1, integration_method: str = "euler") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _v0 = params['desired_speed']
        # _s0 = params['min_gap']
        _tg = params['time_gap']
        _k0 = params['k0']
        _k1 = params['k1']
        _k2 = params['k2']

        num_steps = leader_velocity_traj.shape[0]
        ego_velocity_traj = np.zeros(shape=(num_steps,))
        space_headway_traj = np.zeros(shape=(num_steps,))
        ego_accel_traj = np.zeros(shape=(num_steps-1,))

        ego_velocity_traj[0] = init_ego_velocity
        space_headway_traj[0] = init_space_headway

        for i in range(num_steps-1):
            s_des = self._s0 + self._L + _tg*ego_velocity_traj[i]
            ego_accel_traj[i] = min(
                _k0*(_v0 - ego_velocity_traj[i]),
                _k1*(space_headway_traj[i] - s_des) + _k2*(leader_velocity_traj[i] - ego_velocity_traj[i])
            )
            if self.config.use_acceleration_constraints:
                ego_accel_traj[i] = max(min(ego_accel_traj[i], self.config.max_acceleration), self.config.min_acceleration)
            
            if integration_method == 'euler':
                space_headway_traj[i+1] = space_headway_traj[i] + dt*(leader_velocity_traj[i] - ego_velocity_traj[i])
                ego_velocity_traj[i+1] = ego_velocity_traj[i] + dt*ego_accel_traj[i]
            else:
                raise NotImplementedError
        
        return ego_accel_traj, space_headway_traj, ego_velocity_traj



@dataclass
class ACCModelConfig(CFModelConfig):
    
    k0: float = 0.1
    k1: float = 0.23
    k2: float =  0.07
    time_gap: float = 1.2
    min_gap: float = 2.0
    vehicle_length: float = 5.0
    desired_speed: float = 30
    use_acceleration_constraints: bool = True
    max_acceleration: float = 5.0
    min_acceleration: float = -5.0

    def create(self) -> ACCModel:
        return ACCModel(self)
