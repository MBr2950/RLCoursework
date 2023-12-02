#  basic pseudocode (as guide, won't be efficient):
#  https://moodle.bath.ac.uk/mod/page/view.php?id=1146237&forceview=1

#  pseudocode:
#  https://moodle.bath.ac.uk/mod/page/view.php?id=1146241

import gymnasium as gym
import torch
from torch import nn
import random
env = gym.make('Humanoid-v4', render_mode="human", exclude_current_positions_from_observation=False)
observation, info = env.reset()

D = list()#todo: initialise this list by allowing agent to wander randomly for some time steps 



pi1 = ActorNetwork()
pi2 = ActorNetwork()
q1 = CriticNetwork()
q2 = CriticNetwork()

Beta = 0.1 #Incremental refreshing rate


#Main loop, i is number of episodes
for i in range(1000):
    N = "something"# do gautian noise or something here ant init
    observation, info = env.reset()
    #an episode
    while (True):
        
        #chooses action, notes new information
        action = env.action_space.sample(pi1.ChooseAction(observation, "???????")) + N

        observation, reward, terminated, truncuated, info = env.step(action)
        D.append("{current state??????}", action, reward, observation)

        #updates Action-Value estimate (NN)
        #
        #
        #
        #
        pi1.train()
        # Get each 
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = pi1(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        
        #update the beta thing instead of C here
        #if i % C == 0:
        #    q2.Refresh(q1)
        #
        #
        #




        #ends the episode
        if terminated or truncuated:
            observation, info = env.reset()
            break

env.close()





## The Neural Networks
class ActorNetwork(torch.nn.module):

    def __init__(self):
        #initialises the network
        #we need to decide what our network will look like
        self.NN = nn.Sequential(
            nn.Linear(46, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 32),
            nn.ReLU(),
            nn.Linear(32, 16))


    def ChooseAction(self, State):
        #uses the NN to estimate an action for a given state
        return self.NN(State)
    
        #TODO: Add randomisation
    
    def Update(self):
        #trains the network
        pass

        

    def Refresh():
        #Updates pi2 to match pi1
        pass

class CriticNetwork(torch.nn.module):

    def __init__(self):
        #initialises the network
        self.NN = nn.Sequential(
            nn.Linear(62, 120),
            nn.ReLU(),
            nn.Linear(120, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
            nn.ReLU(),
            nn.Linear(10, 1))
    
    def ActionValue(self, State, Action):
        #uses the NN to estimate the action-value for a given state and action

        return self.NN(State + Action) # + concatentates the 2 lists
    
    def Update(self):
        #trains the network
        pass

    def Refresh():
        #Updates q2 to match q1
        pass