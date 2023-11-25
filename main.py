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

q1 = ActionValueNetwork()
q2 = ActionValueNetwork()
C = 5 #num of loops between updates of q2 from q1


#Main loop, i is number of episodes
for i in range(1000):

    #an episode
    while (True):
        
        #chooses action, notes new information
        action = env.action_space.sample(q1.ChooseAction())

        observation, reward, terminated, truncuated, info = env.step(action)
        D.append("{current state??????}", action, reward, observation)

        #updates Action-Value estimate (NN)
        #
        #
        #
        #

        
        #updates q2
        if i % C == 0:
            q2.Refresh(q1)

        #ends the episode
        if terminated or truncuated:
            observation, info = env.reset()
            break

env.close()





## The Neural Networks
class ActionValueNetwork:

    def __init__(self):
        #initialises the network
        #we need to decide what our network will look like
        pass

    def ChooseAction():
        #uses the NN to estimate an action for a given state
        pass
    
    def Update():
        #trains the network
        pass

    def Refresh(q1):
        #Updates q2 to match q1
        pass
