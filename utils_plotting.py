import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_leader_single_follower(time: np.ndarray, leader_velocity: np.ndarray, ego_velocity: np.ndarray, 
                                space_headway:np.ndarray, acceleration_command:np.ndarray, time_gap_command: np.ndarray) -> None:
    """
    Plots the trajectory of the ego vehicle and leader vehicle
    Parameters:
    -----------
        time: numpy.ndarray, shape=(N,)
        leader_velocity: numpy.ndarray, shape=(N,)
            Trajectory of the velocity of leading vehicle
        ego_velocity: numpy.ndarray, shape=(N,)
            Trajectory of velocity of ego vehicle 
        space_headway: numpy.ndarray, shape=(N,)
            Trajectory of space headway between leader and ego vehicles
        accleration_command: numpy.ndarray, shape=(N-1,)
            Trajectory of acceleration command of the ego vehicle 
        time_gap_command: numpy.ndarray, shape=(N-1,)
            Trajectory of time gap command of the ego vehicle
    Returns:
    --------
        None
    """
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    ax[0, 0].plot(time, leader_velocity, label="Leader")
    ax[0, 0].plot(time, ego_velocity, label="Follower")
    ax[0, 0].set(xlabel="Time (s)", ylabel="Speed (m/s)")
    ax[0, 0].legend()

    ax[0, 1].plot(time, space_headway)
    ax[0, 1].set(xlabel="Time (s)", ylabel="Space Headway (m)")

    ax[1, 0].plot(time[:-1], acceleration_command)
    ax[1, 0].set(xlabel="Time (s)", ylabel="Acceleration (m/s^2)")

    ax[1, 1].plot(time[:-1], time_gap_command)
    ax[1, 1].set(xlabel="Time (s)", ylabel="TIme Gap (s)")

    plt.show()


def plot_calibration_validation(time: np.ndarray, v_leader: np.ndarray, v_data: np.ndarray, s_data: np.ndarray, 
                                a_data: np.ndarray, v_model: np.ndarray, s_model: np.ndarray, a_model: np.ndarray, 
                                fig_title: str = "") -> None:
    fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    ax[0].plot(time, v_leader, label="Leader - Data")
    ax[0].plot(time, v_data, label="Follower - Data")
    ax[0].plot(time, v_model, label="Follower - ACC model")
    ax[0].legend()
    ax[0].set(xlabel="Time (s)", ylabel="Speed (m/s)")

    ax[1].plot(time, s_data, label="openACC Data")
    ax[1].plot(time, s_model, label="ACC calibrated model")
    ax[1].legend()
    ax[1].set(xlabel="Time (s)", ylabel="Space Headway (m)")

    ax[2].plot(time[:-1], a_data, label="openACC Data")
    ax[2].plot(time[:-1], a_model, label="ACC calibrated model")
    ax[2].legend()
    ax[2].set(xlabel="Time (s)", ylabel="Acceleration (m/s^2)")

    fig.suptitle(fig_title)
    plt.show()


def plot_leader_multiple_followers(time: np.ndarray, leader_velocity: np.ndarray, ego_vel_arr: np.ndarray, space_headway_arr: np.ndarray, 
                                   acc_comm_arr: np.ndarray, tg_comm_arr: np.ndarray, fig_title: str = "") -> None:
    F = ego_vel_arr.shape[0]
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    ax[0, 0].plot(time, leader_velocity, label='Leader')
    for i in range(F):
        ax[0, 0].plot(time, ego_vel_arr[i], label=f'F{i+1}')
    ax[0, 0].legend()
    ax[0, 0].set(xlabel='Time (s)', ylabel='Speed (m/s)')

    for i in range(F):
        ax[0, 1].plot(time, space_headway_arr[i], label=f'F{i+1}')
    ax[0, 1].legend()
    ax[0, 1].set(xlabel='Time (s)', ylabel='Space Headway (m)')

    for i in range(F):
        ax[1, 0].plot(time[:-1], acc_comm_arr[i], label=f'F{i+1}')
    ax[1, 0].legend()
    ax[1, 0].set(xlabel='Time (s)', ylabel='Acceleration (m/s^2)')

    for i in range(F):
        ax[1, 1].plot(time, tg_comm_arr[i], label=f'F{i+1}')
    ax[1, 1].legend()
    ax[1, 1].set(xlabel='Time (s)', ylabel='Time Headway (s)')

    fig.suptitle(fig_title)
    #plt.show()


    