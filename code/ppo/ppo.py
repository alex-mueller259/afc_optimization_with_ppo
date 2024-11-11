import csv
import numpy as np
import torch

from actor_critic import ActorCritic
from wt_env import WtEnv


def write_file(file_path, data_list):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(data_list)


class PPOAgent:
    def __init__(self, file_paths, nn_input, env_input, reward_input, gae_input=[0.99, 0.95],
                 agent_input=[4, 5, 0.5, 0.001, 0.2]):
        # file_paths = [file_path_timestep, file_path_episode]
        # nn_input = [num_inputs, num_outputs, hidden_size, lr, std]
        # env_input = [bounds, with_experiment, static_vol_flow, random_init, t_p_init, t_off_init]
        # reward_input = [timestep_reward_type, episode_reward_type, timestep_factor, episode_factor]
        # gae_input = [gamma, tau]
        # agent_input = [ppo_epochs, mini_batch_size, c1, c2, clip_param]

        # Initialize input variables
        self.file_path_timestep, self.file_path_episode = file_paths
        self.timestep_reward_type, self.episode_reward_type, self.timestep_factor, self.episode_factor = reward_input
        self.gamma, self.tau = gae_input
        self.ppo_epochs, self.mini_batch_size, self.c1, self.c2, self.clip_param = agent_input

        # Verify input variables
        assert (self.timestep_reward_type in ['delta_gamma', 'sum_of_gamma', 'gamma_global']), \
            'The parameter timestep_reward_type must be be either "delta_gamma", "sum_of_gamma" or "gamma_global"'
        assert (self.episode_reward_type in ['sum_of_gamma', 'gamma_global']), \
            'The parameter timestep_reward_type must be be either "sum_of_gamma" or "gamma_global"'

        # Initialize environment
        self.env = WtEnv(*env_input)

        # Set the manual seed for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Initialize Neural Network
        self.model = ActorCritic(*nn_input)

        # Initialize ppo variables
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.entropy = 0
        self.returns = []
        self.advantages = []

    def clear_memory(self):
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.entropy = 0

    def store_memory(self, log_prob, value, reward, terminated, state, action):
        self.log_probs.append(log_prob)
        self.values.append(value)
        # self.sum_of_gamma += reward
        # self.rewards.append(torch.FloatTensor([self.sum_of_gamma]).unsqueeze(1).to(self.model.device))
        self.rewards.append(torch.FloatTensor([reward]).unsqueeze(1).to(self.model.device))
        self.masks.append(torch.FloatTensor([1 - terminated]).unsqueeze(1).to(self.model.device))
        self.states.append(state)
        self.actions.append(action)

    def compute_gae(self, next_value):
        values_gae = self.values + [next_value]
        gae = 0
        self.returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values_gae[step + 1] * self.masks[step] - values_gae[step]
            gae = delta + self.gamma * self.tau * self.masks[step] * gae
            self.returns.insert(0, gae + values_gae[step])

        # Concatenates variables for PPO update
        self.returns = torch.cat(self.returns).detach()
        self.values = torch.cat(self.values).detach()
        self.advantages = self.returns[:, 0] - self.values

    def ppo_iter(self):
        batch_size = len(self.states)

        if batch_size > self.mini_batch_size:  # That's usually the case
            n_iter = batch_size // self.mini_batch_size
            mini_batch_size = self.mini_batch_size
        else:  # That's only the case if the episodes terminates before timestep = mini_batch_size
            # In this case just hand over all timesteps that have been run so far
            n_iter = 1
            mini_batch_size = batch_size

        # n_iter = 1

        for i in range(n_iter):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            # rand_ids = range(i, i + mini_batch_size)

            yield torch.FloatTensor(np.array([self.states[i].cpu().numpy() for i in rand_ids])).to(self.model.device), \
                torch.FloatTensor(np.array([self.actions[i].cpu().numpy() for i in rand_ids])).to(self.model.device), \
                torch.FloatTensor(np.array([self.log_probs[i].cpu().detach().numpy() for i in rand_ids])).to(self.model.device), \
                torch.FloatTensor(np.array([self.returns[i].cpu().numpy() for i in rand_ids])).to(self.model.device), \
                torch.FloatTensor(np.array([self.advantages[i].cpu().numpy() for i in rand_ids])).to(self.model.device)

    def ppo_update(self):
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter():

                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp().transpose(0, 1)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage

                actor_loss = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy

                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

        return loss, actor_loss, critic_loss*self.c1, entropy*(-self.c2)

    def learning(self, num_episodes, num_timesteps):

        for i in range(num_episodes):
            print(f'########## Episode {i+1:2.0f} / {num_episodes} ##########')

            # Initialize variables as lists
            self.clear_memory()

            # Reset environment
            state = self.env.reset()

            for j in range(num_timesteps):
                print(f'******* Time Step {j+1:2.0f} / {num_timesteps} *******')

                # Turn 'state' variable into PyTorch Tensor
                state = torch.FloatTensor(state).to(self.model.device)

                # Get NN outputs (dist = Normal distribution of actions to take, value = output of critic NN)
                dist, value = self.model(state)

                # Define new action
                action = dist.sample()  # tensor([dt_p, dt_off]), random sample from dist

                # Pass action to environment
                next_state, reward, terminated = self.env.step(action, self.timestep_reward_type, j)

                # End the episode if t_p or t_off are outside of the boundaries
                if terminated:
                    break

                # Get distribution properties
                log_prob = dist.log_prob(action)
                self.entropy += dist.entropy().mean()

                # Save latest variables to their associated lists
                self.store_memory(log_prob, value, reward, terminated, state, action)

                # Save variables to file
                write_list = [i, j, action.cpu().numpy()[0], action.cpu().numpy()[1]] + self.env.write_list \
                             + [value.cpu().detach().numpy()[0], dist.loc.cpu().detach().numpy()]
                write_file(self.file_path_timestep, write_list)

                state = next_state

            # After last time step:

            # Ignore the episode if it didn't get past the first timestep
            if terminated and j == 0:
                continue

            # Determine episode reward
            if self.episode_reward_type == 'sum_of_gamma':
                episode_reward = self.env.sum_of_gamma / j
            elif self.episode_reward_type == 'gamma_global':
                episode_reward = self.env.rewards_total[-1]

            # Calculate composite reward
            self.rewards = [(self.timestep_factor * timestep_reward) + (self.episode_factor * episode_reward)
                            for timestep_reward in self.rewards]

            # Get GAE (Generalized Advantage Estimation)
            next_state = torch.FloatTensor(next_state).to(self.model.device)
            _, next_value = self.model(next_state)
            self.compute_gae(next_value)

            # Update PPO policy
            loss, actor_loss, critic_loss, entropy = self.ppo_update()

            # Save variables to file
            write_list = [i, episode_reward, self.rewards[-1].cpu().detach().numpy()[0, 0], self.env.rewards_total[-1],
                          loss.cpu().detach().numpy(), actor_loss.cpu().detach().numpy(),
                          critic_loss.cpu().detach().numpy(), entropy.cpu().detach().numpy()]
            write_file(self.file_path_episode, write_list)

            print(f'### End of Episode {i+1:2.0f} / {num_episodes} ###')

        # Save current model
        self.model.save_checkpoint()

