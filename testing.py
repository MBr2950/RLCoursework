import gymnasium as gym
import torch
from torch import nn
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
env = gym.make('Ant-v4', healthy_z_range = (0.5, 1), exclude_current_positions_from_observation=False)
observation, info = env.reset()

## Defining the Actor class
class ActorNetwork(torch.nn.Module):
    def __init__(self):
        """Initialises the Actor network."""
        super(ActorNetwork, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(29, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Linear(32, 8))
        
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
         
## Defining the Critic class
class CriticNetwork(torch.nn.Module):
    def __init__(self):
        """Initialises the Critic network."""
        super(CriticNetwork, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(37, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 45),
            nn.ReLU(),
            nn.Linear(45, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 1))
        
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

# Initialize the Actor and Critic Networks
loaded_pi1 = ActorNetwork()
loaded_q1 = CriticNetwork()

# Load the saved state dictionaries
loaded_pi1.load_state_dict(torch.load('./actor_model.pth'))
loaded_q1.load_state_dict(torch.load('./critic_model.pth'))

# Set the networks to evaluation mode
loaded_pi1.eval()
loaded_q1.eval()

num_episodes = 1000  # Define the number of episodes for evaluation
average_episode_rewards = []  # To store the average reward per episode

for episode in range(num_episodes):
    observation, info = env.reset()
    total_reward = 0

    while True:
        # Choose an action using the loaded actor network
        action = loaded_pi1.ChooseAction(observation)

        # Perform the action in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Check for the end of the episode
        if terminated or truncated:
            break

    # Calculate and record the average reward for this episode
    average_reward = total_reward
    average_episode_rewards.append(average_reward)

# Close the environment
env.close()

# Plotting results
df = pd.DataFrame({'Episode': range(1, num_episodes + 1), 'Average Reward': average_episode_rewards})
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x='Episode', y='Average Reward', data=df).set(title="Evaluation of REINFORCE on Ant-v4 - Average Rewards per Episode")
plt.show()