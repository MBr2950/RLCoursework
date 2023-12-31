# Based off https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py

import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

# Actor neural network for function approximation
class ActorCritic(torch.nn.Module):
    def __init__(self, inputDims, outputDims):
        """Initialises the Actor network."""
        super(ActorCritic, self).__init__()

        layers = [
            nn.Linear(inputDims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        ]

        self.model = torch.nn.Sequential(*layers)

        # Mean and standard deviation of actions are found, for 
        # sampling actions from a normal distribution
        self.modelMean = torch.nn.Sequential(
            torch.nn.Linear(32, outputDims)
        )

        self.modelStdDev = torch.nn.Sequential(
            torch.nn.Linear(32, outputDims)
        )

        self.optimizer = Adam(self.parameters(), lr=0.0003)

        # Avoids type issues
        self.double()

    def forward(self, state):
        """Performs forward pass through network."""
        # Contains the actual values at nodes
        x = self.model(torch.tensor(state))

        # Mean and standard deviation of predicted best action are
        # returned, to allow for sampling from normal distribution
        means = self.modelMean(x)
        epsilon = 1e-6  # Small constant to prevent zero or very small stdDev
        stdDevs = torch.log(1 + torch.exp(self.modelStdDev(x))) + epsilon

        return means, stdDevs

    def update(self, loss):
        """Trains the network."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Carries out the REINFORCE algorithm
class REINFORCE:
    def __init__(self, inputDims, outputDims):
        """Initiates the algorithm."""
        self.inputDims = inputDims # Number of observations 
        self.outputDims = outputDims # Number of actions

        # Hyperparameters set arbitrarily
        self.gamma = 0.95

        self.probabilities = [] # stores the probability of taking a given action
        self.rewards = [] # stores the reward of that action
        self.fullRewards = [] # stores the total rewards of an episode

        # Instance of neural network
        self.network = ActorCritic(inputDims, outputDims)

    def chooseAction(self, state):
        """Samples an action from the distribution according to mean and standard deviation of the predicted best action."""
        means, stdDevs = self.network(state)

        # Defines a distibution for each action, samples from it, then finds the 
        # log probability of taking that action, for calculating loss
        actions = []
        distributions = []
        fullProbability = 0
        
        for i in range(self.outputDims):
            distributions.append(torch.distributions.normal.Normal(means[i], stdDevs[i]))
            actions.append(distributions[i].sample())
            fullProbability += distributions[i].log_prob(actions[i])

        self.probabilities.append(fullProbability / outputDims)

        return actions

    def updateNetwork(self):
        """Updates the network."""
        gradient = 0
        gradients = []
        loss = 0

        # Follow the trajectory in reverse
        for reward in self.rewards[::-1]:
            gradient = reward + (self.gamma * gradient)
            gradients.insert(0, gradient)

        # Calculates the loss of that trajectory
        for i in range(len(self.probabilities)):
            loss += self.probabilities[i].mean() * gradients[i] * (-1)

        # Update network based on this loss
        self.network.update(loss)

        # Reset lists for next epsiode
        self.fullRewards.append(np.array(self.rewards).sum())
        self.probabilities = []
        self.rewards = []

def plot(agent):
    """Creates a graph showing average rewards over time."""
    smoothedRewards = pandas.Series.rolling(pandas.Series(agent.fullRewards), 10).mean()
    smoothedRewards = [element for element in smoothedRewards]
    matplotlib.pyplot.plot(agent.fullRewards)
    matplotlib.pyplot.plot(smoothedRewards)
    matplotlib.pyplot.plot()
    matplotlib.pyplot.xlabel('Episode')
    matplotlib.pyplot.ylabel('Reward')
    matplotlib.pyplot.show()

env = gym.make('Ant-v4', healthy_z_range=(0.5, 1.0), render_mode = "human")

# Seeding to replicate results (using 0, 1, and 2)
np.random.seed(0)
torch.manual_seed(0)
env.reset(seed=4)

# Number of episodes to train for
totalNumEpisodes = 1000
max_steps = 1000
inputDims = env.observation_space.shape[0]
outputDims = env.action_space.shape[0]
totalRewards = 0
rewards_to_plot = []

agent = REINFORCE(inputDims, outputDims)

# Run algorithm for set number of episodes
for episode in range(totalNumEpisodes):
    # Reset environment and keep stepping until episode ends
    state, _ = env.reset()

    terminated = False
    steps = 0
    total_rewards = 0
    while not terminated and steps < max_steps:
        # Choose an action based on the state, observe result, update network
        action = agent.chooseAction(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        agent.rewards.append(reward)
        steps += 1
    #print(steps)

    agent.updateNetwork()

    rewards_to_plot.append(total_rewards)
    print(total_rewards)

    # Print average reward every 100 episodes
    if episode % 10 == 0:
        try:
            averageReward = np.mean(agent.fullRewards[episode - 10:])
        except:
            averageReward = 0
        print("Episode:", episode, "Average Reward:", averageReward)

# Close the environment
env.close()

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
plt.title("Training REINFORCE for Ant-v4")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
