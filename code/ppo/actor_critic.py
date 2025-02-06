import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, learning_rate, save_file, std=0.0001, output_scale=1,
                 deterministic=False):
        super(ActorCritic, self).__init__()
        # Initialize variables
        self.std = std
        self.output_scale = output_scale
        self.save_file = save_file
        self.deterministic = deterministic

        # Define Actor Network
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh()
        )

        # Define Critic Network
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Initialize the weights
        # self.init_weights()

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, maximize=False)

        # Check if NN can be run on GPU
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.to(self.device)

    def forward(self, x):
        value = self.critic(x)

        mu = self.actor(x) * self.output_scale

        if not self.deterministic:
            std = torch.FloatTensor([self.std, self.std]).to(self.device)
            dist = Normal(mu, std)
        else:
            dist = mu

        return dist, value

    def init_weights(self):
        if isinstance(self, nn.Linear):
            nn.init.normal_(self.weight, mean=0., std=0.1)
            nn.init.constant_(self.bias, 0.1)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.save_file)

    def load_checkpoint(self, load_file):
        self.load_state_dict(torch.load(load_file))
