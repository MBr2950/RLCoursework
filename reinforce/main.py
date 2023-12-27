# Based off https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py

import gymnasium as gym
import numpy as np
import torch, pandas, seaborn
import matplotlib.pyplot

# Actor neural network for function approximation
class ActorCritic(torch.nn.Module):
    def __init__(self, inputDims, outputDims):
        """Initialises the Actor network."""
        super(ActorCritic, self).__init__()

        layers = [
            torch.nn.Linear(inputDims, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
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

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

        # Avoids type issues
        self.double()

    def forward(self, state):
        """Performs forward pass through network."""
        # contains the actual values at nodes
        x = self.model(torch.tensor(state))

        # Mean and standard deviation of predicted best action are
        #  returned, to allow for sampling from normal distribution
        means = self.modelMean(x)
        stdDevs = torch.log(1 + torch.exp(self.modelStdDev(x)))

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
        self.gamma = 0.9

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

env = gym.make('Ant-v4', healthy_z_range = (0.5, 1.0), exclude_current_positions_from_observation=False)

# Number of episodes to train for
totalNumEpisodes = 10000
inputDims = env.observation_space.shape[0]
outputDims = env.action_space.shape[0]
totalRewards = []

agent = REINFORCE(inputDims, outputDims)

# Run algorithm for set number of episodes
for episode in range(totalNumEpisodes):
    # Reset environment and keep stepping until episode ends
    state, info = env.reset()

    terminated = False
    while not terminated:
        # Choose an action based on the state, observe result, update network
        action = agent.chooseAction(state)
        state, reward, terminated, truncated, info = env.step(action)
        agent.rewards.append(reward)

    agent.updateNetwork()

    # Print average reward every 100 episodes
    if episode % 100 == 0:
        try:
            averageReward = np.mean(agent.fullRewards[episode - 100:])
        except:
            averageReward = 0
        print("Episode:", episode, "Average Reward:", averageReward)
    # Plot graph every 1000 episodes, plotting stops training
    if episode % 1000 == 0 and episode != 0:
        env.plot()

# Create a new environment, this one will render episodes to show result of training
env = gym.make("Ant-v4", healthy_z_range = (0.5, 1), render_mode = "human")

# Keep running algorithm forever
while True:
    state, info = env.reset()

    terminated = False
    while not terminated:
        action = agent.chooseAction(state)
        state, reward, terminated, truncated, info = env.step(action)
        agent.rewards.append(reward)

    agent.updateNetwork()