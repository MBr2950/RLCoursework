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
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)


    def ChooseAction(self, State):
        #uses the NN to estimate an action for a given state
        return self.NN(State)
    
        #TODO: Add randomisation
    
    def Update(self, values):
        #trains the network
        self.optimizer.zero_grad()
        E = torch.tensor(0)
        for i in range(len(values)):
            E += i * values[i]
        E = round(E)
        E.backward()
        E = -E
        
        self.optimizer.step()
        

     def Refresh(pi):
        #Updates pi2 to match pi1
        for i in range(self.parameters):
            self.parameters = (Beta * pi.parameters) + (1- Beta)*self.parameters

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
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)
    
    def ActionValue(self, State, Action):
        #uses the NN to estimate the action-value for a given state and action

        return self.NN(State + Action) # + concatentates the 2 lists
    
    def Update(self, y1, y2):
        #trains the network
        self.optimizer.zero_grad()
        loss = nn.functional.huber_loss(y1, y2)
        loss.backward()
        self.optimizer.step()
        
    def Refresh(q):
            #Updates q2 to match q1
            for i in range(self.parameters):
                self.parameters = (Beta * q.parameters) + (1- Beta)*self.parameters
    

    
D = list()#todo: initialise this list by allowing agent to wander randomly for some time steps 



pi1 = ActorNetwork()
pi2 = ActorNetwork()
q1 = CriticNetwork()
q2 = CriticNetwork()



Beta = 0.1 #Incremental refreshing rate
minibatch = 5 #Taken from D
gamma = 0.1 #learning rate

#Main loop, i is number of episodes
for i in range(1000):
    N = "something"# do gautian noise or something here ant init
    observation, info = env.reset()
    #an episode
    while (True):
        
        #chooses action, notes new information
        action = env.action_space.sample(pi1.ChooseAction(observation, "???????")) + N

        state = observation
        observation, reward, terminated, truncuated, info = env.step(action)
        D.append(state, action, reward, observation)

        #updates Action-Value estimate (NN)
        #
        #
        #
        #
        for j in range(minibatch):
            transition = D[random.randint(0, len(D))]
            # Picks a random sample from D 
            #0-State, 1- Action, 2- Reward, 3- Observation
            q1y = transition[2] + gamma * q2.ActionValue(transition[3], pi2.ChooseAction(transition[3]))
            q1.Update(q1y, q1.ActionValue(transition[0], transition[1]))

            pi1.Update(q1.ActionValue(transition[0], pi1.ChooseAction(transition[0])))
        #update the beta thing instead of C here
        if i % C == 0:
            pi2.Refresh(pi1)
            q2.Refresh(q1)

        #ends the episode
        if terminated or truncuated:
            observation, info = env.reset()
            break

env.close()







    
       


    
     

