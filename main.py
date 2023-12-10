#  basic pseudocode (as guide, won't be efficient):
#  https://moodle.bath.ac.uk/mod/page/view.php?id=1146237&forceview=1

#  pseudocode:
#  https://moodle.bath.ac.uk/mod/page/view.php?id=1146241

import gymnasium as gym
import torch
from torch import nn
import random
import numpy as np
env = gym.make('Humanoid-v4', exclude_current_positions_from_observation=False)
observation, info = env.reset()

## The Neural Networks
class ActorNetwork(torch.nn.Module):


    def __init__(self):
        #initialises the network
        #we need to decide what our network will look like
        super(ActorNetwork, self).__init__()
        self.NN = nn.Sequential(
            nn.Linear(378, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Linear(32, 17))
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)


    def ChooseAction(self, State):
        #uses the NN to estimate an action for a given state
        State = torch.from_numpy(State).float()
        return (self.NN(State)).detach().numpy()# converts from np to tensor, then back
    
        #TODO: Add randomisation
    
    def Update(self, values):
        #trains the network
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
            param1 = Beta * param1.data + (1-Beta) * param2.data

       #Inspired by:
       #https://github.com/DLR-RM/stable-baselines3/issues/93
       #May need to cite

         

class CriticNetwork(torch.nn.Module):

    def __init__(self):
        #initialises the network
        super(CriticNetwork, self).__init__()
        self.NN = nn.Sequential(
            nn.Linear(395, 900),
            nn.ReLU(),
            nn.Linear(900, 300),
            nn.ReLU(),
            nn.Linear(300, 90),
            nn.ReLU(),
            nn.Linear(90, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 1))
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)
    
    def ActionValue(self, State, Action):
        #uses the NN to estimate the action-value for a given state and action
        return self.NN(torch.from_numpy(np.concatenate((State, Action))).float())
    
    def Update(self, y1, y2):
        #trains the network
        self.optimizer.zero_grad()
        loss = nn.functional.huber_loss(y1, y2)
        loss.backward()
        self.optimizer.step()
        
    def Refresh(self, q):
            #Updates q2 to match q1
            for param1, param2 in zip(self.parameters(), q.parameters()):
                param1 = Beta * param1.data + (1-Beta) * param2.data
    

def DataFunction(reward):
    print("Current reward:")
    print("healthy_reward + forward_reward - ctrl_cost - contact_cost: " + str(reward))
    print("https://gymnasium.farama.org/environments/mujoco/humanoid/#rewards")
    
D = list()#todo: initialise this list by allowing agent to wander randomly for some time steps 
observation, info = env.reset()
for i in range(1000):
    action = env.action_space.sample() #pick random action
    state = observation
    observation, reward, terminated, truncuated, info = env.step(action)
    state = state.astype(float)
    action = action.astype(float)
    observation = observation.astype(float)
    D.append([state, action, reward, observation])

pi1 = ActorNetwork()
pi2 = ActorNetwork()
q1 = CriticNetwork()
q2 = CriticNetwork()



Beta = 0.1 #Incremental refreshing rate
minibatch = 5 #Taken from D
gamma = 0.1 #learning rate
dataprint = 0

#Main loop, i is number of episodes
for i in range(1000):
    observation, info = env.reset()
    #an episode
    while (True):
        
        #chooses action, notes new information
        action = pi1.ChooseAction(observation)


        for i in action:
            i = i + np.random.normal(0, 0.1)#adds randomisation to action for all 16 limbs,+/- 0.1 max, caps at -0.4 and 0.4
            if i > 0.4:
                i = 0.4
            if i < -0.4:
                i = -0.4

        state = observation
        observation, reward, terminated, truncuated, info = env.step(action)
        state = state.astype(float)
        action = action.astype(float)
        observation = observation.astype(float)
        D.append([state, action, reward, observation])

        dataprint += 1
        if dataprint == 100:
            dataprint = 0
            DataFunction(reward)


        #updates Action-Value estimate (NN)
        for j in range(minibatch):
            transition = D[random.randint(0, len(D) - 1)]
            # Picks a random sample from D 
            #0-State, 1- Action, 2- Reward, 3- Observation
            q1y = transition[2] + gamma * q2.ActionValue(transition[3], pi2.ChooseAction(transition[3]))
            q1.Update(q1y, q1.ActionValue(transition[0], transition[1]))

            pi1.Update(q1.ActionValue(transition[0], pi1.ChooseAction(transition[0])))
        

        #update the beta thing instead of C here
        pi2.Refresh(pi1)
        q2.Refresh(q1)

        #ends the episode
        if terminated or truncuated:
            observation, info = env.reset()
            break

env.close()







    
       


    
     

