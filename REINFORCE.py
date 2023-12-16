# Based off https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/#sphx-glr-tutorials-training-agents-reinforce-invpend-gym-v26-py

from torch.distributions.normal import Normal
import gymnasium as gym
import numpy as np
import torch, random
import torch.nn as nn
import torch.optim as optim

# Neural network for function approximation
class ActorCritic(nn.Module):
    def __init__(self, inputDims, outputDims):
        super(ActorCritic, self).__init__()
        layers = [
            nn.Linear(inputDims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        ]

        self.model = nn.Sequential(*layers)

        # Mean and standard deviation of actions are found, for 
        # sampling actions from a normal distribution
        self.modelMean = nn.Sequential(
            nn.Linear(32, outputDims)
        )

        self.modelStdDev = nn.Sequential(
            nn.Linear(32, outputDims)
        )

        # Avoids type issues
        self.double()


    # Performs forward pass to calculate loss
    def forward(self, x):
        # sharedFeatures contains the actual values at nodes
        sharedFeatures = self.model(torch.tensor(x))

        means = self.modelMean(sharedFeatures)
        stdDevs = torch.log(
            1 + torch.exp(self.modelStdDev(sharedFeatures))
        )

        return means, stdDevs



# Carries out the steps of the REINFORCE algorithm
class REINFORCE:
    def __init__(self, inputDims, outputDims):
        # Hyperparameters set arbitrarily
        self.learningRate = 1e-4
        self.gamma = 0.99
        self.episodes = 1e-6

        # Probabilities stores the probability of taking a given action, 
        # rewards stores the reward of that action
        self.probabilities = []
        self.rewards = []

        self.network = ActorCritic(inputDims, outputDims)
        self.optimiser = torch.optim.Adam(self.network.parameters())


    # Samples an action from the distribution according to mean
    #  and standard deviation of the estimated best action
    def chooseAction(self, state):
        means, stdDevs = self.network(state)

        # Defines a distibution, samples from it, then finds the 
        # probability of taking that action, for calculating loss
        distribution = Normal(means[0] + self.episodes, stdDevs[0] + self.episodes)
        action = distribution.sample()
        probability = distribution.log_prob(action)

        action = action.numpy()

        self.probabilities.append(probability)

        return action


    #Updates the network 
    def update(self):
        gradient = 0
        gradients = []

        # Follow the trajectory in reverse
        for reward in self.rewards[::-1]:
            gradient = reward + self.gamma * gradient
            gradients.insert(0, gradient)

        gradients = torch.tensor(gradients)

        # Calculates the loss of that trajectory
        loss = 0
        for i in range(len(self.probabilities)):
            loss += self.probabilities[i].mean() * gradients[i] * (-1)

        # Update network
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Reset lists for next epsiode
        self.probabilities = []
        self.rewards = []



# Create and wrap the environment
env = gym.make("InvertedPendulum-v4")
wrappedEnv = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

totalNumEpisodes = int(5e3)  # Total number of episodes
# Observation-space of InvertedPendulum-v4 (4)
inputDims = env.observation_space.shape[0]
# Action-space of InvertedPendulum-v4 (1)
outputDims = env.action_space.shape[0]
totalRewards = []

for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(inputDims, outputDims)
    rewards = []

    for episode in range(totalNumEpisodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        observation, info = wrappedEnv.reset(seed=seed)

        done = False
        while not done:
            action = agent.chooseAction(observation)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            # print(action)
            observation, reward, terminated, truncated, info = wrappedEnv.step([action])
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        rewards.append(wrappedEnv.return_queue[-1])
        agent.update()

        if episode % 100 == 0:
            avgReward = int(np.mean(wrappedEnv.return_queue))
            print("Episode:", episode, "Average Reward:", avgReward)

    totalRewards.append(rewards)


rewards_to_plot = [[reward[0] for reward in rewards] for rewards in totalRewards]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()