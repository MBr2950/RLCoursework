import gymnasium as gym
import torch
from torch import nn
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque

env = gym.make('Ant-v4', healthy_z_range = (0.5, 1.0))

# Seeding to replicate results (using 0, 1, and 2)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

rewardlist = list()
rewards_to_plot = list()

for i in range(10000):
    observation, _ = env.reset()

    while (True):
        state = observation
        action = np.random.random_sample(size = 8)
        action = (action - 0.5) * 2 # Scales properly (1.0, 1.0)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewardlist.append(reward)

        # Ends the episode
        if terminated or truncated:
            break

    if i == 0:
        avgReward = sum(rewardlist)
        #print("Episode: " + str(i) + ". Reward Avg = " + str(avgReward))
    elif i % 100 == 0:
        avgReward = sum(rewardlist) / 100
        rewards_to_plot.append(avgReward)
        rewardlist = list()
        #print("Episode: " + str(i) + ". Reward Avg = " + str(avgReward))

df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="Training REINFORCE for Ant-v4 - Average Reward per Episode"
)
plt.show()
