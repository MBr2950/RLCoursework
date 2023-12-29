import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta = 0.01 # Incremental refreshing rate

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
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.003, momentum=0.5)

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
            nn.Linear(37, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.003, momentum=0.5)
    
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

def test_model():
    """Loads the trained model and tests it in the render mode."""
    env = gym.make('Ant-v4', render_mode="human")  

    # Initialize the Actor and Critic Networks
    loaded_pi1 = ActorNetwork()
    loaded_q1 = CriticNetwork()

    # Load the saved state dictionaries
    loaded_pi1.load_state_dict(torch.load('RLCoursework/trained_models/ddpg_actor_model.pth'))
    loaded_q1.load_state_dict(torch.load('RLCoursework/trained_models/ddpg_critic_model.pth'))

    # Set the networks to evaluation mode
    loaded_pi1.eval()
    loaded_q1.eval()

    # Run the environment
    state, _ = env.reset()
    while True:
        # Convert state to tensor and move to device
        state = torch.from_numpy(state).float().to(device)

        # Get action from policy
        with torch.no_grad():
            action, _ = loaded_pi1.act(state)
        
        # Step in the environment
        state, _, terminated, truncated, _ = env.step(action.cpu().numpy())

        if terminated or truncated:
            state, _ = env.reset()

if __name__ == "__main__":
    test_model()