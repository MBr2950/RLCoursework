import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """Initialises the Actor-Critic network."""
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """Forward pass for action mean and state value."""
        action_mean = self.actor(state)
        state_value = self.critic(state)
        return action_mean, state_value

    def act(self, state):
        """Selects action and log probability through distribution."""
        # Get action mean
        action_mean, _ = self.forward(state)
        # Calculates covariance matrix (assummes action independence)
        cov_mat = torch.diag(self.log_std.exp().pow(2))
        # Defines distribution to model probabilties of actions
        dist = MultivariateNormal(action_mean, cov_mat)
        # Sample action and probability (used for loss calculation)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        """Evaluate given state and action against policy."""
        # Get action mean and state value
        action_mean, state_value = self.forward(state)
        # Calculate covariance matrix and distribution
        cov_mat = torch.diag(self.log_std.exp().pow(2))
        dist = MultivariateNormal(action_mean, cov_mat)
        # Compute probability of action under distribution
        action_logprobs = dist.log_prob(action)
        # Compute entropy (randomness measure), used for exploration
        dist_entropy = dist.entropy()
    
        return action_logprobs, torch.squeeze(state_value), dist_entropy

def test_model():
    """Loads the trained model and tests it in the render mode."""
    env = gym.make('Ant-v4', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize the network
    policy = ActorCritic(state_dim, action_dim).to(device)

    # Load the saved state dictionary from trained model
    policy.load_state_dict(torch.load('RLCoursework/trained_models/ppo_policy_state.pth'))

    # Set the network in evaluation mode
    policy.eval()

    # Run the environment
    state, _ = env.reset()
    while True:
        # Convert state to tensor and move to device
        state = torch.from_numpy(state).float().to(device)

        # Get action from policy
        with torch.no_grad():
            action, _ = policy.act(state)
        
        # Step in the environment
        state, _, terminated, truncated, _ = env.step(action.cpu().numpy())

        if terminated or truncated:
            state, _ = env.reset()

if __name__ == "__main__":
    test_model()