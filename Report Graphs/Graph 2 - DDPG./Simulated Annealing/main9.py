# Based off  https://moodle.bath.ac.uk/mod/page/view.php?id=1146268
  
import gymnasium as gym
import torch
from torch import nn
import json
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Defining the Actor class
class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        """Initialises the Actor network."""
        super(ActorNetwork, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

    def ChooseAction(self, State):
        """Uses the NN to estimate an action for a given state."""
        # Converts from np to tensor, then back
        State = torch.from_numpy(State).float()
        return (self.NN(State)).detach().numpy()
    
    def Update(self, values):
        """Trains the network."""
        # Where E represents Expected Value, and loss has the correct data type.
        values = values.detach().numpy()
        self.optimizer.zero_grad()
        E = 0
        for i in range(len(values)):
            E += (i + 1) * values[i]
        loss = torch.tensor(E, requires_grad=True)
        loss = -loss
        loss.backward()
        self.optimizer.step()
        

    def Refresh(self, pi):
    #Updates pi2 to match pi1

        #Title:            [enhancement] Polyak Averaging could be done faster
        #Author:           Quinn Sinclair
        #Date:             09/07/20
        #Available From:   https://github.com/DLR-RM/stable-baselines3/issues/93 
        #Access Date:      02/12/23
        for param1, param2 in zip(self.parameters(), pi.parameters()):
            param1.data.mul_(1-beta)
            param1.data.add_(beta*param2.data)

## Defining the Critic class
class CriticNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        """Initialises the Critic network."""
        super(CriticNetwork, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(state_dim+action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)
    
    def ActionValue(self, State, Action):
        """Uses the NN to estimate the action-value for a given state and action."""
        return self.NN(torch.from_numpy(np.concatenate((State, Action))).float())
    
    def Update(self, y1, y2):
        """Trains the network."""
        self.optimizer.zero_grad()
        loss = nn.functional.huber_loss(y1, y2)
        loss.backward()
        self.optimizer.step()
        
    def Refresh(self, q):
        """Updates q2 to match q1."""
        
        #Title:            [enhancement] Polyak Averaging could be done faster
        #Author:           Quinn Sinclair
        #Date:             09/07/20
        #Available From:   https://github.com/DLR-RM/stable-baselines3/issues/93 
        #Access Date:      02/12/23
        for param1, param2 in zip(self.parameters(), q.parameters()):
            param1.data.mul_(1-beta)
            param1.data.add_(beta*param2.data)

# Replay memory- holds all transition information (limited at maxlength):
# Holds queue of multiple: [0-State, 1-Action, 2-Reward, 3-Observation]
D = deque(maxlen=10000)

env = gym.make('Ant-v4', healthy_z_range=(0.5, 1.0), render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 

# Seeding to replicate results (using 0, 1, and 2)
np.random.seed(0)
torch.manual_seed(0)
env.reset(seed=2347)
runIndex = 9

rewards_export = {f'Run {runIndex}': []}
steps_export = {f'Run {runIndex}': []}

# Allows agent to wander randomly for some time steps
observation, info = env.reset()
for i in range(100):
    for j in range(100):
        state = observation
        action = np.random.random_sample(size = 8)
        action = (action - 0.5) * 2 # Scales properly (1.0, 1.0)
        observation, reward, terminated, truncated, _ = env.step(action)
        D.append([state, action, reward, observation])
    observation, info = env.reset()

pi1 = ActorNetwork(state_dim, action_dim).to(device)
pi2 = ActorNetwork(state_dim, action_dim).to(device)
q1 = CriticNetwork(state_dim, action_dim).to(device)
q2 = CriticNetwork(state_dim, action_dim).to(device)

beta = 0.01 # Incremental refreshing rate
minibatch = 1 # Stabilises learning (taken from D)
gamma = 0.95 # Discounting on future rewards
rewards_to_plot = list()
rewards = list()

# Main loop iterating over each episode
for episode in range(1000):
    observation, _ = env.reset()
    total_rewards = 0
    steps = 0

    while (True):
        steps += 1
        # Chooses action, notes new information
        action = pi1.ChooseAction(observation)

        # Adding randomization to the action for all 8 limbs,+/- 0.1 max
        action += np.random.normal(0, 0.3, size=action.shape) * ((1000 - episode) / 1000)

        # Clipping the actions to be within the range [-1.0, 1.0]
        action = np.clip(action, -1.0, 1.0)

        state = observation
        observation, reward, terminated, truncated, info = env.step(action)
        D.append([state, action, reward, observation])
        total_rewards += reward
        rewards.append(reward)

        # Updates Action-Value estimate (NN)
        for j in range(minibatch):
            # Picks a random sample from D 
            # 0-State, 1- Action, 2- Reward, 3- Observation
            transition = D[random.randint(0, len(D) - 1)]

            # Update q1
            q1y = transition[2] + gamma * q2.ActionValue(transition[3], pi2.ChooseAction(transition[3]))
            q1.Update(q1y, q1.ActionValue(transition[0], transition[1]))

            # Update pi1
            pi1.Update(q1.ActionValue(transition[0], pi1.ChooseAction(transition[0])))
        
        # Updates pi2 and q2 to get slightly closer to pi1 and q1
        pi2.Refresh(pi1)
        q2.Refresh(q1)

        # Ends the episode
        if terminated or truncated:
            break

    # Store total rewards for episode
    rewards_to_plot.append(total_rewards)
    rewards_export[f'Run {runIndex}'].append(total_rewards)
    steps_export[f'Run {runIndex}'].append(steps)

    # Calculates avg of rewards for last 100 episodes
    if episode % 10 == 0 and episode != 0:
        print("Episode: " + str(episode) + ". Avg Reward (last 10 episodes)  = " + str(sum(rewards)/100))
        rewards = list()

# Save the trained models
#torch.save(pi1.state_dict(), 'RLCoursework/trained_models/ddpg_actor_model.pth') # Save Actor Model
#torch.save(q1.state_dict(), 'RLCoursework/trained_models/ddpg_critic_model.pth') # Save Critic Model

with open(f'DDPG{runIndex}.json', 'w') as file:
    json.dump(rewards_export, file)

with open(f'DDPG{runIndex}steps.json', 'w') as file:
    json.dump(steps_export, file)

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
plt.title("Training DDPG for Ant-v4")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
