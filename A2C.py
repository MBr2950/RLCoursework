import torch
import gymnasium as gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot
import pandas


class ActorCritic(nn.Module):
    def __init__(self, inputDims, outputDims, learningRate=0.0003):
        super(ActorCritic, self).__init__()
        actorLayers = [
            torch.nn.Linear(inputDims, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        ]

        criticLayers = [
            torch.nn.Linear(inputDims, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
            torch.nn.ReLU(),
        ]

        self.model = torch.nn.Sequential(*actorLayers)

        # Mean and standard deviation of actions are found, for 
        # sampling actions from a normal distribution
        self.modelMean = torch.nn.Sequential(
            torch.nn.Linear(32, outputDims)
        )

        self.modelStdDev = torch.nn.Sequential(
            torch.nn.Linear(32, outputDims)
        )

        self.criticValue = torch.nn.Sequential(*criticLayers)

        # Avoids type issues
        self.double()

    
    def forward(self, state):
        # x contains the actual criticValues at nodes
        # Sometimes input is array, sometimes tensor?
        if type(state) == np.ndarray:
            x = self.model(torch.tensor(state))
        else:
            x = self.model(torch.tensor(state[0]))

        # Mean and standard deviation of predicted best action are
        #  returned, to allow for sampling from normal distribution
        means = self.modelMean(x)
        stdDevs = torch.log(
            1 + torch.exp(self.modelStdDev(x))
        )

        if type(state) == np.ndarray:
            criticValue = self.criticValue(torch.tensor(state))
        else:
            criticValue = self.criticValue(torch.tensor(state[0]))

        return means, stdDevs, criticValue



class A2C():
    def __init__(self, env):
        self.inputDims = env.observation_space.shape[0]
        self.outputDims = env.action_space.shape[0]

        self.actorCritic = ActorCritic(self.inputDims, self.outputDims)
        self.actorCriticOptimiser = optim.Adam(self.actorCritic.parameters(), lr=0.0003)

        self.allLengths = []
        self.averageLengths = []
        self.allRewards = []
        self.entropyTerm = 0

        self.gamma = 0.99
        self.numSteps = 300
        self.maxEpisodes = 3000

    def chooseAction(self, means, stdDevs):
            # Defines a distibution for each action, samples from it, then finds the 
            # probability of taking that action, for calculating loss
            actions = []
            distributions = []
            # Probability of all actions being taken together
            fullProbability = 0

            for i in range(self.outputDims):
                distributions.append(torch.distributions.Normal(means[i], stdDevs[i]))
                actions.append(distributions[i].sample())
                fullProbability += distributions[i].log_prob(actions[i])
            fullProbability = fullProbability / self.outputDims

            # print(actions)

            return actions, fullProbability


    def updateNetwork(self, criticValues, QValues, probabilities):
        #update actor critic
        criticValues = torch.FloatTensor(np.array(criticValues))
        QValues = torch.FloatTensor(QValues)
        probabilities = torch.stack(probabilities)
            
        advantage = QValues - criticValues
        actorLoss = (-probabilities * advantage).mean()
        criticLoss = (0.5 * advantage.pow(2)).mean()

        # print("\n actorLoss: ", actorLoss, "\n criticLoss: ", criticLoss, "\n entropyTerm: ", self.entropyTerm)
        # print("\n criticValues: ", criticValues, "\n QValues: ", QValues)

        actorCriticLoss = actorLoss + criticLoss #+ 0.001 * self.entropyTerm

        self.actorCriticOptimiser.zero_grad()
        actorCriticLoss.backward()
        self.actorCriticOptimiser.step()


    def main(self, env):
        for episode in range(self.maxEpisodes):
            probabilities = []
            criticValues = []
            rewards = []

            state = env.reset()
            for steps in range(self.numSteps):
                means, stdDevs, criticValue = self.actorCritic.forward(state)
                criticValue = criticValue.detach().numpy()

                action, probability = self.chooseAction(means, stdDevs)

                entropy = -(action * np.log(action))
                newState, reward, terminated, truncated, info = env.step(action)

                rewards.append(reward)
                criticValues.append(criticValue)
                probabilities.append(probability)
                self.entropyTerm += entropy
                state = newState
                
                if terminated or steps == self.numSteps - 1:
                    means, stdDevs, QValue = self.actorCritic.forward(newState)
                    QValue = QValue.detach().numpy()
                    self.allRewards.append(np.sum(rewards))
                    self.allLengths.append(steps)
                    self.averageLengths.append(np.mean(self.allLengths[-10:]))
                    if episode % 100 == 0:                    
                        print("episode: {}, reward: {}, total length: {}, average length: {}".format(episode, np.sum(rewards), steps, self.averageLengths[-1]))
                    break
            
            # compute Qvalues
            QValues = np.zeros_like(criticValues)
            for t in reversed(range(len(rewards))):
                QValue = rewards[t] + self.gamma * QValue
                QValues[t] = QValue

            self.updateNetwork(criticValues, QValues, probabilities)

        self.plot()
        
    
    # Plot results
    def plot(self):
        smoothedRewards = pandas.Series.rolling(pandas.Series(self.allRewards), 10).mean()
        smoothedRewards = [element for element in smoothedRewards]
        matplotlib.pyplot.plot(self.allRewards)
        matplotlib.pyplot.plot(smoothedRewards)
        matplotlib.pyplot.plot()
        matplotlib.pyplot.xlabel('Episode')
        matplotlib.pyplot.ylabel('Reward')
        matplotlib.pyplot.show()

        matplotlib.pyplot.plot(self.allLengths)
        matplotlib.pyplot.plot(self.averageLengths)
        matplotlib.pyplot.xlabel('Episode')
        matplotlib.pyplot.ylabel('Episode length')
        matplotlib.pyplot.show()


if __name__ == "__main__":
    env = gym.make("Hopper-v4")
    a2c = A2C(env)
    a2c.main(env)

    env = gym.make("Hopper-v4", render_mode = "human")
    a2c.main(env)