# Based off https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py

import gymnasium as gym
import numpy as np
import torch, pandas, seaborn
import matplotlib.pyplot

# Neural network for function approximation
class ActorCritic(torch.nn.Module):
    def __init__(self, inputDims, outputDims):
        super(ActorCritic, self).__init__()
        # Hyperparameters set arbitrarily
        self.learningRate = 0.0003

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


        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)

        # Avoids type issues
        self.double()


    # Performs forward pass to calculate loss
    def forward(self, state):
        # x contains the actual values at nodes
        x = self.model(torch.tensor(state))

        # Mean and standard deviation of predicted best action are
        #  returned, to allow for sampling from normal distribution
        means = self.modelMean(x)
        stdDevs = torch.log(
            1 + torch.exp(self.modelStdDev(x))
        )

        return means, stdDevs


    # Update values based on the loss of the current policy
    def update(self, loss):
        # print("\n", loss, "\n")
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


# Carries out the steps of the REINFORCE algorithm
class REINFORCE:
    def __init__(self, inputDims, outputDims):
        self.inputDims = inputDims
        self.outputDims = outputDims

        # Hyperparameters set arbitrarily
        self.learningRate = 0.0003
        self.gamma = 0.99

        # probabilities stores the probability of taking a given action, 
        # rewards stores the reward of that action
        # fullRewards stores the total reward of an episode
        self.probabilities = []
        self.rewards = []
        self.fullRewards = []

        self.network = ActorCritic(inputDims, outputDims)


    # Samples an action from the distribution according to mean
    #  and standard deviation of the estimated best action
    def chooseAction(self, state):
        means, stdDevs = self.network(state)
        # print(means, stdDevs)

        # Defines a distibution for each action, samples from it, then finds the 
        # probability of taking that action, for calculating loss
        actions = []
        distributions = []
        # Probability of all actions being taken together
        fullProbability = 0
        
        for i in range(self.outputDims):
            distributions.append(torch.distributions.normal.Normal(means[i], stdDevs[i]))
            actions.append(distributions[i].sample())
            
            fullProbability += distributions[i].log_prob(actions[i])

        self.probabilities.append(fullProbability / outputDims)

        return actions

        
    #Updates the network 
    def updateNetwork(self):
        gradient = 0
        gradients = []
        loss = 0

        # Follow the trajectory in reverse
        for reward in self.rewards[::-1]:
            gradient = reward + (self.gamma * gradient)
            gradients.insert(0, gradient)

        # print(len(self.probabilities), len(gradients))

        # Calculates the loss of that trajectory
        for i in range(len(self.probabilities)):
            loss += self.probabilities[i].mean() * gradients[i] * (-1)

        # Update network based on this loss
        self.network.update(loss)

        # Reset lists for next epsiode
        self.fullRewards.append(np.array(self.rewards).sum())
        self.probabilities = []
        self.rewards = []


# Create and wrap the environment
env = gym.make("Ant-v4", healthy_z_range = (0.5, 1))
# wrappedEnv = gym.wrappers.RecordEpisodeStatistics(env)  # Records episode-reward

totalNumEpisodes = 100000  # Total number of episodes
inputDims = env.observation_space.shape[0]
outputDims = env.action_space.shape[0]
totalRewards = []

agent = REINFORCE(inputDims, outputDims)

for episode in range(totalNumEpisodes):
    # observation, info = wrappedEnv.reset()
    observation, info = env.reset()

    terminated = False
    while terminated == False:
        action = agent.chooseAction(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        agent.rewards.append(reward)
        # print(agent.rewards)

    agent.updateNetwork()
    if episode % 100 == 0:
        try:
            avgReward = np.mean(agent.fullRewards[episode - 100:])
        except:
            avgReward = 0
        print("Episode:", episode, "Average Reward:", avgReward)
    # Plot graph every 1000 episodes, plotting will stop training, disable if you want to leave it to run
    if episode % 1000 == 0 and episode != 0:
        plot(agent)

def plot(agent):
    smoothedRewards = pandas.Series.rolling(pandas.Series(agent.fullRewards), 10).mean()
    smoothedRewards = [element for element in smoothedRewards]
    matplotlib.pyplot.plot(agent.fullRewards)
    matplotlib.pyplot.plot(smoothedRewards)
    matplotlib.pyplot.plot()
    matplotlib.pyplot.xlabel('Episode')
    matplotlib.pyplot.ylabel('Reward')
    matplotlib.pyplot.show()

# Create and wrap the environment
env = gym.make("Ant-v4", healthy_z_range = (0.5, 1), render_mode = "human")
# wrappedEnv = gym.wrappers.RecordEpisodeStatistics(env)  # Records episode-reward

while True:
    observation, info = env.reset()

    terminated = False
    while terminated == False:
        action = agent.chooseAction(observation)

        observation, reward, terminated, truncated, info = env.step(action)
        agent.rewards.append(reward)

    agent.updateNetwork()