#  basic pseudocode (as guide, won't be efficient):
#  https://moodle.bath.ac.uk/mod/page/view.php?id=1146237&forceview=1

#  pseudocode:
#  https://moodle.bath.ac.uk/mod/page/view.php?id=1146241

import gymnasium as gym
import torch
import random
env = gym.make('Humanoid-v4', render_mode="human")
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
class ActorNetwork:

    def __init__(self):
        #initialises the network
        #we need to decide what our network will look like
        pass

    def ChooseAction(State):
        #uses the NN to estimate an action for a given state
        pass
    
    def Update():
        #trains the network
        pass

    def Refresh():
        #Updates pi2 to match pi1
        pass

class CriticNetwork:

    def __init__(self):
        #initialises the network
        #we need to decide what our network will look like
        pass
    
    def ChooseAction(State):
        #uses the NN to estimate an action for a given state
        pass
    
    def Update():
        #trains the network
        pass

    def Refresh():
        #Updates q2 to match q1
        pass