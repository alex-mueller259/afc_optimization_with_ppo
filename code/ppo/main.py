import csv
from datetime import datetime
import glob
import numpy as np
import os

from ppo import PPOAgent
from measurement import init
from measurement import functions


def create_file(file_path, info_list, header_list):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['timestep_reward_type: ' + info_list[0]])
        writer.writerow(['timestep_factor: ' + info_list[1]])
        writer.writerow(['episode_reward_type: ' + info_list[2]])
        writer.writerow(['episode_factor: ' + info_list[3]])
        writer.writerow(['tp/toff_init: ' + info_list[4]])
        writer.writerow(['std: ' + info_list[5]])
        writer.writerow(['num_episodes: ' + info_list[6]])
        writer.writerow(['num_timesteps: ' + info_list[7]])
        writer.writerow(['model_reload: ' + info_list[8]])
        writer.writerow([])
        writer.writerow(header_list)


def get_latest_file(suffix, file_ending):
    file_list = glob.glob(os.path.join(save_dir, (suffix + '*' + file_ending)))
    return file_list[-1]


if __name__ == '__main__':

    """Define file paths"""

    date_time = datetime.now().strftime('%Y-%m-%d_%H%M')
    save_dir = r'C:\Users\alex-\Documents\03-Uni\00_MASTERARBEIT\RL_TSB\save_dir\random_init'
    file_path_timestep = os.path.join(save_dir, ('timestep_data_' + date_time + '.txt'))
    file_path_episode = os.path.join(save_dir, ('episode_data_' + date_time + '.txt'))

    file_paths = [file_path_timestep, file_path_episode]

    """Neural Network settings"""

    # Hyper params
    num_inputs = 8  # Size of input layers
    num_outputs = 2  # Size of output layers
    output_scale = 1  # Factor to scale the output to ms
    hidden_size = 256  # number of hidden layers
    lr = 3e-4  # learning rate

    std = 0.2  # (ms) standard deviation

    nn_save_file = os.path.join(save_dir, 'model_' + date_time + '.pth')  # File path for model saving

    nn_input = [num_inputs, num_outputs, hidden_size, lr, nn_save_file, std, output_scale]

    """Environmental settings"""

    with_experiment = False  # Select how to get Gamma (with or without live experiment)
    static_vol_flow = False  # Select if the data is obtained with static volume flow (True) or static pressure (False)
    random_init = False  # Whether to select the initial t_p and t_off values randomly or use given values

    # Boundaries of the t_p / t_off values
    t_p_lower = 1
    t_p_upper = 25
    t_off_lower = 1
    t_off_upper = 25
    bounds = [t_p_lower, t_p_upper, t_off_lower, t_off_upper]

    # Initial t_p / t_off values
    t_p_init = 15  # (ms)
    t_off_init = 15  # (ms)

    env_input = [bounds, std, with_experiment, static_vol_flow, random_init, t_p_init, t_off_init]

    """Agent settings"""

    # Set up iteration params
    num_episodes = 1000
    num_timesteps = 15

    # For reward calculation
    timestep_reward_type = 'delta_gamma'  # Must be either 'delta_gamma', 'sum_of_gamma' or 'gamma_global'
    episode_reward_type = 'gamma_global'  # Must be either 'sum_of_gamma' or 'gamma_global'
    timestep_factor = 1
    episode_factor = 0

    reward_input = [timestep_reward_type, episode_reward_type, timestep_factor, episode_factor]

    # For PPO Update
    mini_batch_size = 5
    ppo_epochs = 4
    c1 = 0.5
    c2 = 0.001
    clip_param = 0.2

    agent_input = [ppo_epochs, mini_batch_size, c1, c2, clip_param]

    # For advantage calculation
    gamma = 0.99
    tau = 0.95

    gae_input = [gamma, tau]

    # Initialize PPO agent
    agent = PPOAgent(file_paths, nn_input, env_input, reward_input, gae_input, agent_input)

    # Model reload
    model_reload = input('Do you want to reload the model? (y/n):')
    if model_reload == 'y':
        nn_load_file = get_latest_file('model', '.pth')
        latest_model = input('Do you want to reload the latest model? (y/n):\n' + nn_load_file + '\n')
        if not latest_model == 'y':
            nn_load_file = input('Please provide the file path of the model you want to reload:\n')
            assert os.path.exists(nn_load_file), ('The model file ' + nn_load_file + ' does not exist.')
        agent.model.load_checkpoint(nn_load_file)

    """Create files"""
    if random_init:
        info_init = 'random'
    else:
        info_init = f'[{t_p_init}, {t_off_init}]'

    if model_reload == 'y':
        info_reload = nn_load_file
    else:
        info_reload = 'no'

    info_list = [timestep_reward_type, str(timestep_factor), episode_reward_type, str(episode_factor), info_init,
                 str(std), str(num_episodes), str(num_timesteps), info_reload]
    timestep_list = ['episode', 'timestep', 'dt_p', 'dt_off', 't_p', 't_off', 'reward', 'gamma_global', 'delta_gamma',
                     'sum_of_gamma', 'sum_of_gamma_norm', 'critic_value', 'mu']
    episode_list = ['episode', 'episode_reward', 'reward_last_timestep', 'gamma_global_last_timestep', 'loss',
                    'actor_loss', 'critic_loss', 'entropy']

    create_file(file_path_timestep, info_list, timestep_list)
    create_file(file_path_episode, info_list, episode_list)

    """Other initializations"""

    # If getting Gamma from live experiment, initialize sensors etc.
    if with_experiment:
        # Initialize sensors
        init.init_sensors()

        # Get sensor offset for wall-shear-stress sensors
        print('Offset Measurement for Wall-Shear-Stress Sensors')
        functions.read_signal(init.Tau)
        init.Tau.offset = np.mean(init.Tau.data_in, axis=1)
        print(init.Tau.offset)

        # Wait for wind tunnel to be started
        input('Turn on wind tunnel. Wait for the velocity to reach 20m/s and then type any key to continue')

    """Reinforcement Learning"""

    print('Start learning...')
    agent.learning(num_episodes, num_timesteps)

