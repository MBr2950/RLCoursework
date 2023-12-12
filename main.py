# Basic code largely adapted from the following tutorial:
# https://gymnasium.farama.org/tutorials/training_agents/reinforce_invpend_gym_v26/

import random, cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym



# Neural network for giving actions
class ActorNetwork(nn.Module):
    # observationDim, actionDim are integers defining the number of observations 
    # and actions available in the given model
    def __init__(self, observationDim, actionDim) -> None:
        super().__init__()

        # Define neural network properties
        self.sharedNet = nn.Sequential(
                nn.Linear(observationDim, 500),
                nn.ReLU(),
                nn.Linear(500, 100),
                nn.ReLU(),
                nn.Linear(100, 32),
                nn.ReLU(),
        )

        # Policy Mean specific Linear Layer
        self.policyMeanNet = nn.Sequential(
            nn.Linear(32, actionDim)
        )

        # Policy Std Dev specific Linear Layer
        self.policyStdDevNet = nn.Sequential(
            nn.Linear(32, actionDim)
        )


    # Performs one forward pass on the neural network
    # Takes in a tensor 'x'
    # Returns two tensors, 'actionMeans' and 'actionStdDevs', both predicted
    # from the normal distribution
    def forward(self, x):
        sharedFeatures = self.sharedNet(x.float())

        actionMeans = self.policyMeanNet(sharedFeatures)
        actionStdDevs = torch.log(
            1 + torch.exp(self.policyStdDevNet(sharedFeatures))
        )

        return actionMeans, actionStdDevs


class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = 1e-4  # Learning rate for policy optimization
        self.gamma = 0.99  # Discount factor
        self.eps = 1e-6  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = ActorNetwork(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []



# Create and wrap the environment
env = gym.make("InvertedPendulum-v4")
wrappedEnv = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

numEpisodes = int(1e3)  # Total number of episodes
observationDims = env.observation_space.shape[0]
actionDims = env.action_space.shape[0]
seedRewards = []

for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed
    agent = REINFORCE(observationDims, actionDims)
    reward_over_episodes = []

    for episode in range(numEpisodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrappedEnv.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrappedEnv.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrappedEnv.return_queue[-1])
        agent.update()

        if episode % 100 == 0:
            avg_reward = int(np.mean(wrappedEnv.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)

    seedRewards.append(reward_over_episodes)
    

env = gym.make("InvertedPendulum-v4", render_mode="human")
wrappedEnv = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

numEpisodes = int(1e3)  # Total number of episodes

for seed in [1, 2, 3, 5, 8]:  # Fibonacci seeds
    # set seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Reinitialize agent every seed

    for episode in range(numEpisodes):
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrappedEnv.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrappedEnv.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_over_episodes.append(wrappedEnv.return_queue[-1])
        agent.update()

        if episode % 100 == 0:
            avg_reward = int(np.mean(wrappedEnv.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)
        
            # use mode 'rgb_array'
            # img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            # cv2.imshow("test", img)
            # cv2.waitKey(50)

    seedRewards.append(reward_over_episodes)


rewards_to_plot = [[reward[0] for reward in rewards] for rewards in seedRewards]
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()