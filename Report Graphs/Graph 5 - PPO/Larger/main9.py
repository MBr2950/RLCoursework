import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.distributions import MultivariateNormal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """Initialises the Actor-Critic network."""
        super(ActorCritic, self).__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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

# PPO Agent
class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, K_epochs, eps_clip):
        """Initialises PPO algorithm."""
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, state):
        """Selects an action using old policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        return action.cpu().numpy(), action_logprob.cpu().numpy()

    def update(self, memory):
        """Updates the policy (calculating the loss) using epochs for stability."""
        rewards = list()
        discounted_reward = 0

        # Calculating discounted rewards (including terminated states)
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Prepare tensors for rewards, old states, old actions, and old action probabilities 
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        # Update the policy k times (adding stability)
        for _ in range(self.K_epochs):
            # Evaluate old states and actions using policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Loss calculation
            # Computing ratio between probabilities (exp used to cancel log)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Computing advantages
            advantages = rewards - torch.squeeze(state_values).detach()
            # Computing loss
            surr1 = ratios * advantages
            # Clipping loss at epsilon value stabilises update
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # Total loss combines surrogate loss, value loss, and entropy loss (for exploration)
            loss = -torch.min(surr1, surr2) + 0.5 * (state_values - rewards).pow(2) - 0.01 * dist_entropy

            # Parameters update
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Update old policy to new policy (for next round of training)
        self.policy_old.load_state_dict(self.policy.state_dict())

# Memory class
class Memory:
    def __init__(self):
        """Initialises the memory class."""
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        """Clears memory (used at every policy update)."""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def train(env):
    """Trains the PPO algorithm on the environment."""
    # Hyperparameters
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    learning_rate = 0.003 
    gamma = 0.9
    epochs = 3 # stabilises the policy calculation
    eps_clip = 0.2 # limits policy update 
    episodes = 3000 # max training episodes
    update_timestep = 2048 # update policy every n timesteps

    ppo = PPO(state_dim, action_dim, learning_rate, gamma, epochs, eps_clip)
    memory = Memory()

    timestep = 0
    rewards_to_plot = list()
    rewards = list()

    for episode in range(episodes):
        state, _ = env.reset()
        total_rewards = 0
        steps = 0

        while True:
            timestep += 1
            steps += 1

            # Running on the old policy
            action, logprob = ppo.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total_rewards += reward
            rewards.append(reward)

            # Saving states, actions, probabilities, rewards, and terminal
            memory.states.append(torch.tensor(state, dtype=torch.float32))
            memory.actions.append(torch.tensor(action))
            memory.logprobs.append(torch.tensor(logprob))
            memory.rewards.append(reward)
            memory.is_terminals.append(terminated or truncated)

            # Update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            if terminated or truncated:
                break

        # Store total rewards for episode
        rewards_to_plot.append(total_rewards)
        rewards_export[f'Run {runIndex}'].append(total_rewards)
        steps_export[f'Run {runIndex}'].append(steps)

        # Calculates avg of rewards for last 100 episodes
        if episode % 10 == 0 and episode != 0:
            print("Episode: " + str(episode) + ". Avg Reward (last 10 episodes)  = " + str(sum(rewards)/10))
            rewards = list()

    # Save the trained model
    torch.save(ppo.policy.state_dict(), f'model{runIndex}.pt')

    return rewards_to_plot

if __name__ == '__main__':
    env = gym.make('Ant-v4', render_mode="human") # Start environment

    #Seeding to replicate results (using 0, 1, and 2)
    runIndex = 9
    np.random.seed(0)
    torch.manual_seed(0)
    env.reset(seed=99999) #765
    rewards_export = {f'Run {runIndex}': []}
    steps_export = {f'Run {runIndex}': []}

    # Train and store average rewards
    rewards_to_plot = train(env)

    env.close() # Close environment

    with open(f'PPO{runIndex}.json', 'w') as file:
        json.dump(rewards_export, file)

    with open(f'PPO{runIndex}steps.json', 'w') as file:
        json.dump(steps_export, file)

    # Plotting results
    # Calculate total and rolling rewards in DataFrame
    df = pd.DataFrame({'Episode': range(1, len(rewards_to_plot) + 1), 'Total Reward': rewards_to_plot})
    df['Rolling Avg Reward'] = df['Total Reward'].rolling(window=100).mean()
    # Plot total and rolling rewards
    sns.set(style="darkgrid", context="talk", palette="rainbow")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Episode', y='Total Reward', data=df, label='Total Reward', color='blue')
    sns.lineplot(x='Episode', y='Rolling Avg Reward', data=df, label='Average Reward (100 episodes)', color='red')
    # Set plot title and labels
    plt.title("Training PPO for Ant-v4")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
