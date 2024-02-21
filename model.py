import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
from dataclasses import dataclass


class CFModel(ABC):
    def __init__(self):
        super(CFModel, self).__init__()
    
    @abstractmethod
    def step_using_parameters_from_dict(self, params: dict, 
                                        leader_velocity: float, ego_velocity: float, space_headway: float, 
                                        integration_method: str = "euler") -> Tuple[float, float, float]:
        pass
    
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
    def set_parameters_from_dict(self, params:dict) -> None:
        pass


@dataclass
class CFModelConfig(ABC):

    @abstractmethod
    def create(self) -> CFModel:
        pass