#  pseudocode:
#  https://moodle.bath.ac.uk/mod/page/view.php?id=1146268
  
import gymnasium as gym
import torch
from torch import nn
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque

env = gym.make('Ant-v4', healthy_z_range = (0.5, 1.0), exclude_current_positions_from_observation=False)
observation, info = env.reset()

## Defining the Actor class
class ActorNetwork(torch.nn.Module):
    def __init__(self):
        """Initialises the Actor network."""
        super(ActorNetwork, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(29, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

    def ChooseAction(self, State):
        """Uses the NN to estimate an action for a given state."""
        # Converts from np to tensor, then back
        State = torch.from_numpy(State).float()
        return (self.NN(State)).detach().numpy()
    
    def Update(self, values):
        """Trains the network."""
        # Where E and E1 represent Expected Value (E is working value, E1 has correct datatype)
        values = values.detach().numpy()
        self.optimizer.zero_grad()
        E = 0
        for i in range(len(values)):
            E += (i + 1) * values[i]
        E1 = torch.tensor(E, requires_grad=True)
        E1.backward()
        E1 = -E1
        self.optimizer.step()
        

    def Refresh(self, pi):
        #Updates pi2 to match pi1
        for param1, param2 in zip(self.parameters(), pi.parameters()):
            param1 = beta * param1.data + (1-beta) * param2.data

       #Inspired by:
       #https://github.com/DLR-RM/stable-baselines3/issues/93

         
## Defining the Critic class
class CriticNetwork(torch.nn.Module):
    def __init__(self):
        """Initialises the Critic network."""
        super(CriticNetwork, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(37, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)
    
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
        for param1, param2 in zip(self.parameters(), q.parameters()):
            param1 = beta * param1.data + (1-beta) * param2.data

# Replay memory- holds all transition information (limited at maxlength):
# Holds queue of multiple: [0-State, 1-Action, 2-Reward, 3-Observation]
D = deque(maxlen=100000)

# Allows agent to wander randomly for some time steps
observation, info = env.reset()
for i in range(100):
    state = observation
    action = np.random.random_sample(size = 8)
    action = (action - 0.5) * 2 # Scales properly (1.0, 1.0)
    observation, reward, terminated, truncated, info = env.step(action)
    D.append([state, action, reward, observation])

pi1 = ActorNetwork()
pi2 = ActorNetwork()
q1 = CriticNetwork()
q2 = CriticNetwork()

beta = 0.01 # Incremental refreshing rate
minibatch = 5 # Taken from D
gamma = 0.9 # Discounting on future rewards
dataprint = 0
rewardlist = list()
rewards_to_plot = list()

# Main loop iterating over each episode
for i in range(1000):
    observation, info = env.reset()

    while (True):
        # Chooses action, notes new information
        action = pi1.ChooseAction(observation)

        # Adding randomization to the action for all 8 limbs,+/- 0.1 max
        action += np.random.normal(0, 0.1, size=action.shape)

        # Clipping the actions to be within the range [-1.0, 1.0]
        action = np.clip(action, -1.0, 1.0)

        state = observation
        observation, reward, terminated, truncated, info = env.step(action)
        D.append([state, action, reward, observation])
        rewardlist.append(reward)

        # Updates Action-Value estimate (NN)
        for j in range(minibatch):
            # Picks a random sample from D 
            # 0-State, 1- Action, 2- Reward, 3- Observation
            transition = D[random.randint(0, len(D) - 1)]

            # Train q1
            q1y = transition[2] + gamma * q2.ActionValue(transition[3], pi2.ChooseAction(transition[3]))
            q1.Update(q1y, q1.ActionValue(transition[0], transition[1]))

            # Train pi1
            pi1.Update(q1.ActionValue(transition[0], pi1.ChooseAction(transition[0])))
        
        # Updates pi2 and q2 to get slightly closer to pi1 and q1
        pi2.Refresh(pi1)
        q2.Refresh(q1)

        # Ends the episode
        if terminated or truncated:
            observation, info = env.reset()
            break
    
    # Calculates avg of rewards for last 100 episodes
    if i == 0:
        avgReward = sum(rewardlist)
        print("Episode: " + str(i) + ". Reward Avg = " + str(avgReward))
    elif i % 100 == 0:
        avgReward = sum(rewardlist) / 100
        rewards_to_plot.append(avgReward)
        rewardlist = list()
        print("Episode: " + str(i) + ". Reward Avg = " + str(avgReward))

# Save the trained models
torch.save(pi1.state_dict(), 'actor_model.pth') # Save Actor Model
torch.save(q1.state_dict(), 'critic_model.pth') # Save Critic Model

env.close()

# Plotting results
df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="Training REINFORCE for Ant-v4 - Average Reward per Episode"
)
plt.show()