import numpy as np
import scipy as sp
import json

from dataclasses import dataclass
from typing import Tuple, List, Optional

from utils_plotting import plot_leader_single_follower

class VTG_ACC:
    def __init__(self, config: "VTG_ACC_Config") -> None:
        self._config = config
        self._k0 = config.k0 
        self._k1 = config.k1 
        self._k2 = config.k2 
        self._v0 = config.desired_speed 
        self._s0 = config.still_distance 
        self._Lveh = config.vehicle_length 
        self._des_tg = config.desired_time_gap
        self._use_acc_cnst = config.use_acceleration_constraints
        if self._use_acc_cnst:
            if config.use_MFC_constraints:
                # TODO: Implement MFC acceleration constraints
                # min_acc and max_acc will be function of velocity (due to powertrain capabilities)
                raise NotImplementedError
            else:
                self._min_acc = config.min_acceleration
                self._max_acc = config.max_acceleration
        self._gamma = config.hinf_bound
        self._rho_s = config.spacing_weight
        self._rho_v = config.velocity_weight
        self._rho_u = config.vtg_weight
        self._A = np.array([[0, -1], [self._k1, -self._k1*self._des_tg-self._k2]])
        self._B_veq = np.array([[1, 0], [self._k2, -self._k1]])
        self._C = np.array([[self._rho_s, 0], [0, self._rho_v], [0, 0]])
        self._R = np.array([[self._gamma**2, 0], [0, -self._rho_u**2]])
        self._Q = self._C.T @ self._C
        self._P = np.empty(shape=(2, 2))
    
    def get_acceleration_command(self, leader_velocity: float, ego_velocity: float,
                                 space_headway: float, vel_eq: Optional[float] = None,
                                 compute_P: bool = True) -> Tuple[float, float]:
        if vel_eq is None:
            vel_eq = leader_velocity

        if compute_P:
            try:
                B = self._B_veq * np.array([[1, 1], [1, vel_eq]])
                self._P = sp.linalg.solve_continuous_are(self._A, B, self._Q, -self._R)
                assert(np.all(np.linalg.eigvals(self._P) > 0))
                s_err = space_headway - self._s0 - self._Lveh - self._des_tg*vel_eq
                v_err = ego_velocity - vel_eq
                g2_x = np.array([[0, -self._k1*vel_eq - self._k1*v_err]])
                ctrl_tg = (-g2_x @ self._P @ np.array([[s_err], [v_err]]))[0, 0]
            except sp.linalg.LinAlgError:
                print(f'Scipy.LinAlgError at {vel_eq}!')
                ctrl_tg = 0.0
        else:
            s_err = space_headway - self._s0 - self._Lveh - self._des_tg*vel_eq
            v_err = ego_velocity - vel_eq
            g2_x = np.array([[0, -self._k1*vel_eq - self._k1*v_err]])
            ctrl_tg = (-g2_x @ self._P @ np.array([[s_err], [v_err]]))[0, 0]
        time_gap = min(max(self._des_tg + ctrl_tg, 0.1), 6)
        desired_space_headway = self._s0 + self._Lveh + time_gap * ego_velocity
        acc_comm = min(
            self._k0 * (self._v0 - ego_velocity),
            self._k1 * (space_headway - desired_space_headway) + self._k2 * (leader_velocity - ego_velocity) 
        )
        if self._use_acc_cnst:
            acc_comm = min(max(self._min_acc, acc_comm), self._max_acc)
        # collision avoidance
        if (ego_velocity**2 - leader_velocity**2) / (2.0*space_headway) >= np.abs(self._min_acc):
            acc_comm = self._min_acc
        return time_gap, acc_comm
    
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
                            plot_trajectory: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                time_gap_trajectory: numpy.ndarray, shape=(N-1,)
                    time gap command of the ego vehicle after time step dt
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
        tg_traj = np.zeros(shape=(N-1,))
        v_traj = np.zeros(shape=(N,))
        s_traj = np.zeros(shape=(N,))
        s_traj[0], v_traj[0] = init_space_headway, init_ego_velocity
        for t in range(N-1):
            tg_traj[t], acc_traj[t] = self.get_acceleration_command(leader_velocity_traj[t], v_traj[t], s_traj[t])
            s_traj[t+1], v_traj[t+1] = self.step(leader_velocity_traj[t], 
                                                 v_traj[t], s_traj[t], acc_traj[t])
        if plot_trajectory:
            plot_leader_single_follower(time, leader_velocity_traj, v_traj, s_traj, 
                                        acc_traj, tg_traj)
        return tg_traj, acc_traj, s_traj, v_traj
    
    def step_using_parameters_from_dict(self, params: dict, 
                                        leader_velocity: float, ego_velocity: float, space_headway: float,
                                        validate: bool = False) -> Tuple[float, float, float]:
        dt = self._config.integration_dt
        if validate:
            try:
                k0, k1, k2 = params['k0'].value, params['k1'].value, params['k2'].value
                s0, Lv, des_tg = params['s0'].value, params['veh_length'].value, params['desired_time_gap'].value
                v0 = params['v0'].value
                rho_s, rho_v, rho_u = params['spacing_weight'].value, params['velocity_weight'].value, params['vtg_weight'].value
                gamma = params['hinf_bound'].value
            except KeyError:
                print("Some Parameters for VTG ACC model are missing!")
                exit(1)
        else:
            try:
                k0, k1, k2 = params['k0'], params['k1'], params['k2']
                s0, Lv, des_tg = params['s0'], params['veh_length'], params['desired_time_gap']
                v0 = params['v0']
                rho_s, rho_v, rho_u = params['spacing_weight'], params['velocity_weight'], params['vtg_weight']
                gamma = params['hinf_bound']
            except KeyError:
                print("Some Parameters for VTG ACC model are missing!")
                exit(1)
        # VTG ACC
        vel_eq = leader_velocity
        A = np.array([[0, -1], [k1, -k1*des_tg-k2]])
        B = np.array([[1, 0], [k2, -k1*vel_eq]])
        C = np.array([[rho_s, 0], [0, rho_v], [0, 0]])
        R = np.array([[gamma**2, 0], [0, -rho_u**2]])
        Q = C.T @ C
        try:
            P = sp.linalg.solve_continuous_are(A, B, Q, -R)
            s_err = space_headway - s0 - Lv - des_tg*vel_eq
            v_err = ego_velocity - vel_eq
            g2_x = np.array([[0, -k1*vel_eq-k1*v_err]])
            ctrl_tg = (-g2_x @ P @ np.array([[s_err], [v_err]]))[0, 0]
        except sp.linalg.LinAlgError:
            ctrl_tg = 0.0
        time_gap = min(max(des_tg + ctrl_tg, 0.1), 6)
        desired_space_headway = s0 + Lv + time_gap * ego_velocity
        acc_comm = k1 * (space_headway - desired_space_headway) + k2 * (leader_velocity - ego_velocity) 
        if self._use_acc_cnst:
            acc_comm = min(max(self._min_acc, acc_comm), self._max_acc)

        if self._config.integration_method == "euler":
            v_next = ego_velocity + acc_comm * dt
            s_next = space_headway + (leader_velocity - ego_velocity) * dt
            return acc_comm, s_next, v_next
        elif self._config.integration_method == "rk4":
            k1_v, k2_v, k3_v, k4_v = acc_comm
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
            return acc_comm, s_next, v_next
        else:
            raise NotImplementedError


@dataclass
class VTG_ACC_Config:
    k0: float # gain for velocity error (speed control mode)
    k1: float # gain for spacing error (gap control mode)
    k2: float # gain for relative velocity (gap control mode)
    desired_time_gap: float  # time gap setting in car-following regime
    desired_speed: float # desired speed in free-flow regime
    still_distance: float = 3.0 # still distance in car-following regime
    vehicle_length: float = 5.0 # length of the vehicle
    use_acceleration_constraints: bool = True
    use_MFC_constraints: bool = False
    min_acceleration: float = -7
    max_acceleration: float = 5
    integration_method: str = "euler" # Integration method, available = "euler" or "rk4"
    integration_dt: float = 0.1 # Time step for integration
    hinf_bound: float = 0.95 # gamma; bound for H-infinity gain from disturbance to output
    spacing_weight: float = 0.2 # rho_s; penalty weight on spacing for definition of z in H-infinity performance specs
    velocity_weight: float = 0.3 # rho_v; penalty weight on velocity for definition of z in H-infinity performance specs
    vtg_weight: float = 1 # rho_u; penalty weight on VTG control signal for definition of z in H-infinity performance specs

    def create(self, json_file_path: Optional[str] = None) -> VTG_ACC:
        if json_file_path is not None:
            with open(json_file_path, 'r') as file:
                params = json.load(file)
            self.k0, self.k1, self.k2 = params['k0'], params['k1'], params['k2']
            self.desired_speed = params['desired_speed']
            self.still_distance = params['still_distance']
            self.vehicle_length = params['vehicle_length']
            self.desired_time_gap = params['time_gap']
            self.use_acceleration_constraints = params['use_acceleration_constraints']
            self.integration_dt = params['integration_dt']
            if self.use_acceleration_constraints:
                self.min_acceleration = params['min_acceleration']
                self.max_acceleration = params['max_acceleration']
        return VTG_ACC(self)
            