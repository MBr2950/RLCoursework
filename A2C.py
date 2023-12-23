# Based off https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f

import torch, pandas, matplotlib.pyplot
import gymnasium as gym
import numpy as np


# Neural network for function approximation
class ActorCritic(torch.nn.Module):
    def __init__(self, inputDims, outputDims, learningRate=0.0003):
        super(ActorCritic, self).__init__()
        # Hyperparameters set arbitrarily
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

        self.critic = torch.nn.Sequential(*criticLayers)

        # Avoids type issues
        self.double()

    
    def forward(self, state):
        # x contains the actual values at nodes
        x = self.model(torch.tensor(state))

        # Mean and standard deviation of predicted best action are
        #  returned, to allow for sampling from normal distribution
        means = self.modelMean(x)
        stdDevs = torch.log(
            1 + torch.exp(self.modelStdDev(x))
        )

        # Critic simply returns predicted value of a state
        Qvalue = self.critic(torch.tensor(state))

        return means, stdDevs, Qvalue



# Carries out the steps of the A2C algorithm
class A2C():
    def __init__(self, env):
        # Number of observations
        self.inputDims = env.observation_space.shape[0]
        # Number of actions
        self.outputDims = env.action_space.shape[0]

        self.actorCritic = ActorCritic(self.inputDims, self.outputDims)
        self.actorCriticOptimiser = torch.optim.Adam(self.actorCritic.parameters(), lr=0.0003)

        # Rewards per step in episode
        self.allRewards = []
        # Average reward over one episode
        self.averageRewards = []

        # Hyperparameters set arbitrarily
        # Discount value
        self.gamma = 0.99
        # For calculating GAE
        # self.lambdaValue = 0.95

        # Number of episodes to train for
        self.totalNumEpisodes = 100000


    def chooseAction(self, means, stdDevs):
            # Defines a distibution for each action, samples from it, then finds the 
            # probability of taking that action, for calculating loss

            # If outputDims = 1, means[0] changes meaning from first action to first element in action
            if self.outputDims == 1:
                distribution = torch.distributions.Normal(means, stdDevs)
                action = distribution.sample()
                probability = distribution.log_prob(action)
                # Entropy is a measure of randomnessm can help mitigate variance
                entropy = distribution.entropy()

                return action, probability, entropy
            else:
                actions = []
                distributions = []
                # Probability of all actions being taken together
                fullProbability = 0
                entropy = 0

                for i in range(self.outputDims):
                    distributions.append(torch.distributions.Normal(means[i], stdDevs[i]))
                    actions.append(distributions[i].sample())
                    fullProbability += distributions[i].log_prob(actions[i])
                    entropy += distributions[i].entropy()
                fullProbability = fullProbability / self.outputDims
                entropy = entropy / self.outputDims

                return actions, fullProbability, entropy


    # Calculate the loss, then update the network based on this
    def updateNetwork(self, QValues, rewards, probabilities, entropies, terminateds):
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

        self.actorCriticOptimiser.zero_grad()
        actorCriticLoss.backward()
        self.actorCriticOptimiser.step()


    # Main training loop
    def main(self, env):
        # Runs algorithm for given number of episodes
        for episode in range(self.totalNumEpisodes):
            probabilities = []
            entropies = []
            QValues = []
            rewards = []
            terminateds = []

            # Resets the environment, then runs until terminated or truncated (gone on too long)
            state, info = env.reset()
            terminated = False
            truncated = False
            while terminated == False and truncated == False:
                # Chooses an action, then steps using that action and records all information
                means, stdDevs, Qvalue = self.actorCritic.forward(state)
                Qvalue = Qvalue.detach().numpy()
                action, probability, entropy = self.chooseAction(means, stdDevs)
                newState, reward, terminated, truncated, info = env.step(action)

                rewards.append(reward)
                QValues.append(Qvalue)
                probabilities.append(probability)
                entropies.append(entropy)
                terminateds.append(terminated)
                state = newState
                
                # If episode is over, perform final updates, and calculate avertage reward
                if terminated == True or truncated == True:
                    means, stdDevs, QValue = self.actorCritic.forward(newState)
                    QValue = QValue.detach().numpy()
                    self.allRewards.append(np.sum(rewards))
                    self.averageRewards.append(np.mean(self.allRewards[-10:]))

                    # Every 100 episodes print out average reward
                    if episode % 100 == 0:                    
                        print("episode: {}, reward: {}, average reward: {}".format(episode, np.sum(rewards), self.averageRewards[-1]))
                    # Plot graph every 1000 episodes, plotting will stop training, disable if you want to leave it to run
                    if episode % 1000 == 0 and episode != 0:
                        self.plot()
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


if __name__ == "__main__":
    env = gym.make("Ant-v4", healthy_z_range = (0.5, 1))
    a2c = A2C(env)
    a2c.main(env)

    env = gym.make("Ant-v4", healthy_z_range = (0.5, 1), render_mode = "human")
    a2c.main(env)