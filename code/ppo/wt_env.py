import numpy as np
import os
import scipy

from measurement import functions


class WtEnv:
    def __init__(self, bounds, std, with_experiment=False, static_vol_flow=False, random_init=True, t_p_init=None,
                 t_off_init=None):
        # Initialize input variables
        self.t_p_lower, self.t_p_upper, self.t_off_lower, self.t_off_upper = bounds
        self.std = std
        self.t_p_init = t_p_init
        self.t_off_init = t_off_init

        # Whether to obtain Gamma through live experiment or measurement data
        self.with_experiment = with_experiment
        # Whether to keep the volume flow or the pressure of the AFC actuators constant during measurement 
        self.static_vol_flow = static_vol_flow

        # Whether to select the initial t_p and t_off values randomly or use given values
        self.random_init = random_init

        # Make sure that t_p_init and t_off_init are defined when selecting random_init = False
        if not self.random_init:
            assert isinstance(self.t_p_init, (int, float)) and isinstance(self.t_off_init, (int, float)), \
                'Define t_p_init and t_off_init when selecting random_init=False.'

        # Initialize other variables
        self.actions_total = []
        self.rewards_total = []
        self.sum_of_gamma = 0
        self.write_list = []

        self.num_timesteps = None
        self.asymptote = None

    def get_obs_reward(self, t_p, t_off):
        # Select which function to use to get Gamma
        if self.with_experiment:
            gamma_global, gamma_local = functions.measurement_workflow(t_p, t_off, self.static_vol_flow)
        else:
            gamma_global, gamma_local = interp_measurement_data(t_p, t_off)

        observation = gamma_local
        reward = gamma_global

        return observation, reward

    def reset(self):
        self.actions_total = []
        self.rewards_total = []
        self.sum_of_gamma = 0

        if self.random_init:
            # Add / subtract 3*self.std (standard deviation) to make sure that the first time step is always within the
            # given t_p / t_off boundaries
            self.t_p_init = np.random.uniform(low=(self.t_p_lower + 3.5*self.std), high=(self.t_p_upper - 3.5*self.std))
            self.t_off_init = np.random.uniform(low=(self.t_off_lower + 3.5*self.std), high=(self.t_off_upper - 3.5*self.std))

        # Get observation and info
        observation, gamma_global = self.get_obs_reward(self.t_p_init, self.t_off_init)

        self.actions_total.append([self.t_p_init, self.t_off_init])
        self.rewards_total.append(gamma_global)

        return observation

    def step(self, action, reward_type, timestep):
        # Get last t_p and t_off values
        action_total_old = self.actions_total[-1]

        # Derive total values of t_p and t_off
        action_total = [(action_total_old[0] + action.cpu().numpy()[0]),
                        (action_total_old[1] + action.cpu().numpy()[1])]
        print(f'dt_p = {action.cpu().numpy()[0]:.2f}ms   |   dt_off = {action.cpu().numpy()[1]:.2f}ms')
        print(f't_p = {action_total[0]:.2f}ms   |   t_off = {action_total[1]:.2f}ms')

        # Check if t_p and t_off are within the given boundaries
        if (action_total[0] < self.t_p_lower) or (action_total[0] > self.t_p_upper) \
                or (action_total[1] < self.t_off_lower) or (action_total[1] > self.t_off_upper):
            # Set observation and total reward (gamma_global) to 0
            observation = np.zeros(8)
            gamma_global = self.rewards_total[-1]
            terminated = True
        else:
            # Get observation and total reward (gamma_global)
            observation, gamma_global = self.get_obs_reward(action_total[0], action_total[1])
            terminated = False

        delta_gamma = gamma_global - self.rewards_total[-1]
        self.sum_of_gamma += gamma_global
        sum_of_gamma_norm = self.sum_of_gamma / (timestep + 1)

        if reward_type == 'delta_gamma':
            reward = delta_gamma
        elif reward_type == 'sum_of_gamma':
            reward = sum_of_gamma_norm
        elif reward_type == 'gamma_global':
            reward = gamma_global

        print(f'Gamma = {gamma_global:.5f}   |   reward = {reward:.5f}')

        # Store new values
        self.actions_total.append(action_total)
        self.rewards_total.append(gamma_global)
        self.write_list = [action_total[0], action_total[1], reward, gamma_global, delta_gamma, self.sum_of_gamma,
                           sum_of_gamma_norm]

        return observation, reward, terminated


def interp_measurement_data(t_p, t_off):
    # Load data
    save_dir = r'C:\Users\alex-\Documents\03-Uni\00_MASTERARBEIT\RL_TSB\ema2-2024\data\files'
    file_path_coarse = 'const_vol-flow_1-27ms_in2ms.txt'
    file_path_fine = 'const_vol-flow_2-20ms_in1ms.txt'

    data_coarse = np.loadtxt(os.path.join(save_dir, file_path_coarse), skiprows=1)
    data_fine = np.loadtxt(os.path.join(save_dir, file_path_fine), skiprows=1)

    # Rescale data from s to ms
    data_coarse[:, 0:2] = data_coarse[:, 0:2] * 1000
    data_fine[:, 0:2] = data_fine[:, 0:2] * 1000

    # Get upper and lower bounds of fine grid
    tp_l = np.min(data_fine[:, 0])  # lower t_p bound
    tp_u = np.max(data_fine[:, 0])  # upper t_p bound
    toff_l = np.min(data_fine[:, 1])  # lower t_off bound
    toff_u = np.max(data_fine[:, 1])  # upper t_off bound

    # Choose data for interpolation
    if (t_p >= tp_l) & (t_p <= tp_u) & (t_off >= toff_l) & (t_off <= toff_u):
        data = data_fine
    else:
        data = data_coarse

    # Reshape arrays to grid
    data_tp = np.unique(data[:, 0])
    data_toff = np.unique(data[:, 1])

    len_tp = len(data_tp)
    len_toff = len(data_toff)

    data_gamma_global = np.reshape(data[:, 4], (len_tp, len_toff))

    # Interpolate
    gamma_global = scipy.interpolate.interpn((data_tp, data_toff), data_gamma_global, (t_p, t_off), method='cubic',
                                             bounds_error=False, fill_value=None)
    if gamma_global < 0:
        gamma_global = 0

    gamma_local = np.zeros(8)
    for i in range(8):
        data_gamma_local = np.reshape(data[:, (5 + i)], (len_tp, len_toff))
        gamma_local[i] = scipy.interpolate.interpn((data_tp, data_toff), data_gamma_local, (t_p, t_off), method='cubic',
                                                   bounds_error=False, fill_value=None)
        if gamma_local[i] < 0:
            gamma_local[i] = 0

    return gamma_global[0], gamma_local
