# Based off https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

import torch, pandas, matplotlib.pyplot
import json
from torch import nn
from torch.optim import Adam
import gymnasium as gym
import numpy as np

# Actor-Critic neural network for function approximation
class ActorCritic(torch.nn.Module):
    def __init__(self, inputDims, outputDims):
        """Initialises the Actor network."""
        super(ActorCritic, self).__init__()
        
        actorLayers = [
            nn.Linear(inputDims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        ]

        criticLayers = [
            nn.Linear(inputDims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ]

        self.model = torch.nn.Sequential(*actorLayers)

        self.critic = torch.nn.Sequential(*criticLayers)

        # Mean and standard deviation of actions are found, for 
        # sampling actions from a normal distribution
        self.modelMean = torch.nn.Sequential(
            torch.nn.Linear(32, outputDims)
        )

        self.modelStdDev = torch.nn.Sequential(
            torch.nn.Linear(32, outputDims)
        )

        self.optimizer = Adam(self.parameters(), lr=0.0002)

        # Avoids type issues
        self.double()
    
    def forward(self, state):
        """Performs forward pass through network."""
        # contains the actual values at nodes
        x = self.model(torch.tensor(state))

        # Mean and standard deviation of predicted best action are
        #  returned, to allow for sampling from normal distribution
        means = self.modelMean(x)
        epsilon = 1e-6  # Small constant to prevent zero or very small stdDev
        stdDevs = torch.log(1 + torch.exp(self.modelStdDev(x))) + epsilon

        # Critic simply returns predicted value of a state
        Qvalue = self.critic(torch.tensor(state))

        return means, stdDevs, Qvalue

# Carries out the steps of the A2C algorithm
class A2C():
    def __init__(self, env):
        """Initiates the algorithm."""
        self.inputDims = env.observation_space.shape[0] # Number of observations
        self.outputDims = env.action_space.shape[0] # Number of actions

        self.actorCritic = ActorCritic(self.inputDims, self.outputDims)

        self.allRewards = [] # Rewards per step in episode
        self.averageRewards = [] # Average reward over one episode

        # Hyperparameters set arbitrarily
        self.gamma = 0.9 # Discount value
        self.totalNumEpisodes = 10000 # Number of episodes
        # self.lambdaValue = 0.95 # For calculating GAE

    def chooseAction(self, means, stdDevs):
        """Defines a distibution for each action, samples from it, then finds the probability of taking that action, for calculating loss."""
        # If outputDims = 1, means[0] changes meaning from first action to first element in action
        if self.outputDims == 1:
            distribution = torch.distributions.Normal(means, stdDevs)
            action = distribution.sample()
            probability = distribution.log_prob(action)
            entropy = distribution.entropy() # Entropy is a measure of randomness which helps mitigate variance

            return action, probability, entropy
        else:
            actions = []
            distributions = []
            fullProbability = 0 # Probability of all actions being taken together
            entropy = 0

            for i in range(self.outputDims):
                distributions.append(torch.distributions.Normal(means[i], stdDevs[i]))
                actions.append(distributions[i].sample())
                fullProbability += distributions[i].log_prob(actions[i])
                entropy += distributions[i].entropy()

            fullProbability = fullProbability / self.outputDims
            entropy = entropy / self.outputDims

            return actions, fullProbability, entropy

    def updateNetwork(self, QValues, rewards, probabilities, entropies, terminateds):
        """Updates the network."""
        # Calculate returns
        returns = np.zeros_like(QValues)
        myReturn = 0
        for t in reversed(range(len(rewards))):
            myReturn = rewards[t] + self.gamma * myReturn
            returns[t] = myReturn

        QValues = torch.FloatTensor(np.array(QValues))
        returns = torch.FloatTensor(returns)
        probabilities = torch.stack(probabilities)
        entropies = torch.tensor(entropies)
            
        advantages = returns - QValues
        # advantages = self.calculateGAE(rewards, QValues, terminateds)

        # Calculate loss
        actorLoss = (-probabilities * advantages).mean()
        criticLoss = (0.5 * advantages.pow(2)).mean()
        entropy = entropies.mean()
        actorCriticLoss = actorLoss + criticLoss + 0.001 * entropy

        self.actorCritic.optimizer.zero_grad()
        actorCriticLoss.backward()
        self.actorCritic.optimizer.step()

    def main(self, env):
        """Trains the network."""
        # Runs algorithm for given number of episodes
        for episode in range(self.totalNumEpisodes):
            probabilities = []
            entropies = []
            QValues = []
            rewards = []
            terminateds = []
            steps = 0

            # Resets the environment, then runs until terminated or truncated
            state, _ = env.reset()
            terminated = False
            truncated = False

            while not terminated and not truncated:
                # Chooses an action, then steps using that action and records all information
                means, stdDevs, Qvalue = self.actorCritic.forward(state)
                Qvalue = Qvalue.detach().numpy()
                action, probability, entropy = self.chooseAction(means, stdDevs)
                newState, reward, terminated, truncated, _ = env.step(action)

                #print(reward)
                reward *= 1.5
                
                rewards.append(reward)
                QValues.append(Qvalue)
                probabilities.append(probability)
                entropies.append(entropy)
                terminateds.append(terminated)
                state = newState
                steps += 1
                
                # If episode is over, perform final updates, and calculate avertage reward
                if terminated == True or truncated == True or steps > max_steps:
                    means, stdDevs, QValue = self.actorCritic.forward(newState)
                    QValue = QValue.detach().numpy()
                    self.allRewards.append(np.sum(rewards))
                    self.averageRewards.append(np.mean(self.allRewards[-10:]))
                    print(np.sum(rewards)/1.5)
                    rewards_export[f'Run {runIndex}'].append((np.sum(rewards)/1.5))
                    steps_export[f'Run {runIndex}'].append(steps)

                    #x = input()
                    # Every 100 episodes print out average reward
                    if episode % 10 == 0:                    
                        print("episode: {}, reward: {}, average reward: {}".format(episode, np.sum(rewards), self.averageRewards[-1]))
                    break
                
            self.updateNetwork(QValues, rewards, probabilities, entropies, terminateds)

        self.plot()

    # Tried to implement a generalised advantage estimate (GAE) version of the advantage, but empirically
    # this seemed to produce worse results, so TD advantage is used instead
    # def calculateGAE(self, rewards, QValues, terminateds):
    #     T = len(rewards) - 1
    #     gaes = [0] * T
    #     future_gae = 0
    #     for t in reversed(range(T)):
    #         delta = rewards[t] + self.gamma * QValues[t + 1] * (not terminateds[t]) - QValues[t]
    #         gaes[t] = future_gae = delta + self.gamma * self.lambdaValue * (not terminateds[t]) * future_gae
        
    #     return torch.tensor(gaes)

    def plot(self):
        """Plots results."""
        smoothedRewards = pandas.Series.rolling(pandas.Series(self.allRewards), 10).mean()
        smoothedRewards = [element for element in smoothedRewards]
        matplotlib.pyplot.plot(self.allRewards)
        matplotlib.pyplot.plot(smoothedRewards)
        matplotlib.pyplot.plot()
        matplotlib.pyplot.xlabel('Episode')
        matplotlib.pyplot.ylabel('Reward')
        matplotlib.pyplot.show()

if __name__ == "__main__":
    env = gym.make("Ant-v4", healthy_z_range = (0.5, 1), render_mode = "human", exclude_current_positions_from_observation=False)

    runIndex = 7
    max_steps = 1000
    np.random.seed(0)
    torch.manual_seed(0)
    env.reset(seed=777) #2
    rewards_export = {f'Run {runIndex}': []}
    steps_export = {f'Run {runIndex}': []}

    a2c = A2C(env)
    a2c.main(env)

    # Close the environment
    env.close()

    with open(f'A2C{runIndex}.json', 'w') as file:
        json.dump(rewards_export, file)

    with open(f'A2C{runIndex}steps.json', 'w') as file:
        json.dump(steps_export, file)
