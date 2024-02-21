import numpy as np
import json

from dataclasses import dataclass
from typing import Tuple, Optional

from model import CFModel, CFModelConfig
from utils_plotting import plot_leader_single_follower

class VTG_SMC(CFModel):
    def __init__(self, config: "VTG_SMC_Config") -> None:
        super(VTG_SMC, self).__init__()
        self._config = config
        self._k0 = config.k0 
        self._v0 = config.desired_speed 
        self._s0 = config.still_distance 
        self._T = config.T
        self._G = config.G
        self._lam = config.lam
        self._Lveh = config.vehicle_length
        self._use_acc_cnst = config.use_acceleration_constraints
        if self._use_acc_cnst:
            if config.use_MFC_constraints:
                # TODO: Implement MFC acceleration constraints
                # min_acc and max_acc will be function of velocity (due to powertrain capabilities)
                raise NotImplementedError
            else:
                self._min_acc = config.min_acceleration
                self._max_acc = config.max_acceleration
    
    def get_acceleration_command(self, leader_velocity: float, 
                                 ego_velocity: float, space_headway: float) -> float:
        desired_space_headway = self._s0 + self._T * ego_velocity + self._G * (ego_velocity**2) 
        gain_den = (self._T + 2.0 * self._G * ego_velocity)
        acc_comm = min(
            self._k0 * (self._v0 - ego_velocity),
            self._lam * (space_headway - desired_space_headway) / gain_den + (leader_velocity - ego_velocity) / gain_den # gap control mode
        )
        if self._use_acc_cnst:
            acc_comm = min(max(self._min_acc, acc_comm), self._max_acc)
        # collision avoidance
        if (ego_velocity**2 - leader_velocity**2) / (2.0*space_headway) >= np.abs(self._min_acc):
            acc_comm = self._min_acc
        return acc_comm
    
    def step(self, leader_velocity: float, ego_velocity: float, space_headway: float, acc_command: float) -> Tuple[float, float]:
        """
            Performs a single integration step of the ACC model
            Parameters:
            -----------
                leader_velocity: float
                    Current velocity of leading vehicle
                ego_velocity: float
                    Current velocity of ego vehicle
                space_headway: float
                    Current space headway between leader and ego vehicles
            Returns:
            --------
                next_space_headway: float
                    space headway between leader and ego vehicles after time step dt
                next_ego_velocity: float
                    velocity of ego vehicle after time step dt
        """
        dt = self._config.integration_dt
        if self._config.integration_method == "euler":
            v_next = ego_velocity + acc_command * dt
            s_next = space_headway + (leader_velocity - ego_velocity) * dt
            return s_next, v_next
        elif self._config.integration_method == "rk4":
            k1_v, k2_v, k3_v, k4_v = acc_command
            k1_s = leader_velocity - ego_velocity
            v_k1 = ego_velocity + k1_v * dt / 2
            s_k1 = space_headway + k1_s * dt / 2
            
            k2_s = leader_velocity - v_k1
            v_k2 = ego_velocity + k2_v * dt / 2
            s_k2 = space_headway + k2_s * dt / 2

            k3_s = leader_velocity - v_k2
            v_k3 = ego_velocity + k3_v * dt
            s_k3 = space_headway + k3_s * dt

            k4_s = leader_velocity - v_k3

            v_next = ego_velocity + (k1_v + 2*k2_v + 2*k3_v + k4_v) * dt / 6
            s_next = space_headway + (k1_s + 2*k2_s + 2*k3_s + k4_s) * dt / 6
            return s_next, v_next
        else:
            raise NotImplementedError
    
    def simulate_trajectory(self, leader_velocity_traj: np.ndarray, 
                            init_ego_velocity: float, init_space_headway: float,
                            time: Optional[np.ndarray] = None,
                            plot_trajectory: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Simulates the trajectory of the ego vehicle using the ACC model
            Parameters:
            -----------
                leader_velocity_trajectory: numpy.ndarray, shape=(N,)
                    Trajectory of the velocity of leading vehicle
                init_ego_velocity: float
                    Initial velocity of ego vehicle
                init_space_headway: float
                    Initial space headway between leader and ego vehicles
                time: Optional, numpy.ndarray, shape=(N,)
                plot_trajectory: bool, default=True
                    flag of whether to plot the simulated trajectories
            Returns:
            --------
                accleration_trajectory: numpy.ndarray, shape=(N-1,)
                    acceleration command of the ego vehicle after time step dt
                space_headway_trajectory: numpy.ndarray, shape=(N,)
                    space headway between leader and ego vehicles after time step dt
                ego_velocity_trajectory: numpy.ndarray, shape=(N,)
                    velocity of ego vehicle after time step dt
        """
        N = leader_velocity_traj.shape[0]
        if time is None:
            time = np.arange(start=0, step=self._config.integration_dt, stop=N*self._config.integration_dt)
        acc_traj = np.zeros(shape=(N-1,))
        v_traj = np.zeros(shape=(N,))
        s_traj = np.zeros(shape=(N,))
        s_traj[0], v_traj[0] = init_space_headway, init_ego_velocity
        for t in range(N-1):
            acc_traj[t] = self.get_acceleration_command(leader_velocity_traj[t], v_traj[t], s_traj[t])
            s_traj[t+1], v_traj[t+1] = self.step(leader_velocity_traj[t], 
                                                 v_traj[t], s_traj[t], acc_traj[t])
        if plot_trajectory:
            plot_leader_single_follower(time, leader_velocity_traj, v_traj, s_traj, 
                                        acc_traj, self._tg*np.ones(shape=(N-1,)))
        return acc_traj, s_traj, v_traj
    
    def set_parameters_from_dict(self, params: dict) -> None:
        return super().set_parameters_from_dict(params)
    
    def step_using_parameters_from_dict(self, params: dict, leader_velocity: float, ego_velocity: float, space_headway: float, integration_method: str = "euler") -> Tuple[float]:
        return super().step_using_parameters_from_dict(params, leader_velocity, ego_velocity, space_headway, integration_method)



@dataclass
class VTG_SMC_Config(CFModelConfig):
    k0: float # gain for velocity error (speed control mode)
    desired_speed: float # desired speed in free-flow regime
    lam: float = 2.0 # sliding surface gain for spacing error (gap control mode)
    T: float = 0.0019 # linear coefficient for spacing policy (gap control mode)
    G: float = 0.0448  # quadratic coefficient for spacing policy (gap control mode)
    still_distance: float = 3.0 # still distance in car-following regime
    vehicle_length: float = 5.0 # length of the vehicle
    use_acceleration_constraints: bool = True
    use_MFC_constraints: bool = False
    min_acceleration: float = -7
    max_acceleration: float = 5
    integration_method: str = "euler" # Integration method, available = "euler" or "rk4"
    integration_dt: float = 0.1 # Time step for integration

    def create(self, json_file_path: Optional[str] = None) -> VTG_SMC:
        if json_file_path is not None:
            with open(json_file_path, 'r') as file:
                params = json.load(file)
            self.k0 = params['k0']
            self.desired_speed = params['desired_speed']
            self.still_distance = params['still_distance']
            self.vehicle_length = params['vehicle_length']
            self.use_acceleration_constraints = params['use_acceleration_constraints']
            self.integration_dt = params['integration_dt']
            if self.use_acceleration_constraints:
                self.min_acceleration = params['min_acceleration']
                self.max_acceleration = params['max_acceleration']
        return VTG_SMC(self)