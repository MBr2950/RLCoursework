#  pseudocode:
#  https://moodle.bath.ac.uk/mod/page/view.php?id=1146268

  
import gymnasium as gym
import torch
from torch import nn
import random
import numpy as np
import matplotlib
env = gym.make('Ant-v4', exclude_current_positions_from_observation=False)
observation, info = env.reset()

## The Neural Networks
class ActorNetwork(torch.nn.Module):


    def __init__(self):
        #initialises the network
        #we need to decide what our network will look like
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
        #Where E and E1 represent Expected Value (E is working value, E1 has correct datatype)
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
    


D = list()#Replay memory- holds all transition information:
          #Holds list of multiple: [0-State, 1-Action, 2-Reward, 3-Observation]

observation, info = env.reset()
for i in range(10):
    state = observation
    action = np.random.random_sample(size = 8)
    action = (action - 0.5) * 0.8 #scales properly
    observation, reward, terminated, truncuated, info = env.step(action)
    D.append([state, action, reward, observation])
#allows agent to wander randomly for some time steps 


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
    rewardlist = list()

    while (True):
        
        #chooses action, notes new information
        action = pi1.ChooseAction(observation)

        #adds randomisation to action for all 16 limbs,+/- 0.1 max, caps at -0.4 and 0.4
        for j in action:
            j = j + np.random.normal(0, 0.1)
            if j > 0.4:
                j = 0.4
            if j < -0.4:
                j = -0.4

        state = observation
        observation, reward, terminated, truncuated, info = env.step(action)
        D.append([state, action, reward, observation])

        #updates Action-Value estimate (NN)
        for j in range(minibatch):
            # Picks a random sample from D 
            #0-State, 1- Action, 2- Reward, 3- Observation
            transition = D[random.randint(0, len(D) - 1)]

            #train q1
            q1y = transition[2] + gamma * q2.ActionValue(transition[3], pi2.ChooseAction(transition[3]))
            q1.Update(q1y, q1.ActionValue(transition[0], transition[1]))

            #train pi1
            pi1.Update(q1.ActionValue(transition[0], pi1.ChooseAction(transition[0])))
        

        #updates pi2 and q2 to get slightly closer to pi1 and q1
        pi2.Refresh(pi1)
        q2.Refresh(q1)

        #ends the episode
        if terminated or truncuated:
            observation, info = env.reset()
            break
    
    #calculates avg of rewards for an episode
    avgReward = 0
    for rew in rewardlist:
        avgReward += rew
    avgReward = avgReward / len(rewardlist)

    print("Episode: " + str(i) + ". Reward Avg = " + str(avgReward))

env.close()

df1 = pd.DataFrame(rewards_to_plot).melt()
df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
sns.set(style="darkgrid", context="talk", palette="rainbow")
sns.lineplot(x="episodes", y="reward", data=df1).set(
    title="REINFORCE for InvertedPendulum-v4"
)
plt.show()

