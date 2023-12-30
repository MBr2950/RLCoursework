import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta = 0.01 # Incremental refreshing rate

## Defining the Actor class
class ActorNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        """Initialises the Actor network."""
        super(ActorNetwork, self).__init__()

        self.NN = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

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

def test_model():
    """Loads the trained model and tests it in the render mode."""
    env = gym.make('Ant-v4', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]  

    # Initialize the Actor network
    loaded_pi1 = ActorNetwork(state_dim, action_dim)

    # Load the saved state dictionaries
    loaded_pi1.load_state_dict(torch.load('RLCoursework/trained_models/ddpg_actor_model.pth'))

    # Set the network to evaluation mode
    loaded_pi1.eval()

    # Run the environment
    state, _ = env.reset()
    while True:
        # Get action from policy
        with torch.no_grad():
            action = loaded_pi1.ChooseAction(state)

        # Step in the environment
        state, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            state, _ = env.reset()

if __name__ == "__main__":
    test_model()