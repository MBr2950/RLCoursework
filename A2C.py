# Neural network for function approximation
class ActorCritic(nn.Module):
    def __init__(self, inputDims, outputDims):
        super(ActorCritic, self).__init__()
        
        actorLayers = [
            nn.Linear(inputDims, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        ]

        # Mean and standard deviation of actions are found, for 
        # sampling actions from a normal distribution
        self.actorMean = nn.Sequential(
            nn.Linear(32, outputDims)
        )

        self.actorStdDev = nn.Sequential(
            nn.Linear(32, outputDims)
        )

        criticLayers = [
            nn.Linear(inputDims, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        ]

        # Actor and critic have different outputs (distribution vs loss)
        self.actor = nn.Sequential(*actorLayers)
        self.critic = nn.Sequential(*criticLayers)

        self.actorOptimiser = torch.optim.Adam(self.actor.parameters())
        self.criticOptimiser = torch.optim.Adam(self.critic.parameters())


        # Avoids type issues
        self.double()


    # Performs forward pass to calculate loss
    def forward(self, state):
        # sharedFeatures contains the actual values at nodes
        sharedFeatures = self.actor(torch.tensor(state))

        means = self.actorMean(sharedFeatures)
        stdDevs = torch.log(
            1 + torch.exp(self.actorStdDev(sharedFeatures))
        )

        criticValue = self.critic(torch.tensor(state))

        return means, stdDevs, criticValue


    def get_losses(
        self,
        rewards: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        entropy: torch.Tensor,
        masks: torch.Tensor,
        gamma: float,
        lam: float,
        ent_coef: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        T = len(rewards)
        advantages = torch.zeros(T, self.n_envs, device=device)

        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rewards[t] + gamma * masks[t] * value_preds[t + 1] - value_preds[t]
            )
            gae = td_error + gamma * lam * masks[t] * gae
            advantages[t] = gae

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)
    

    def updateNetwork(self, actorLoss, criticLoss):
        self.actorOptimiser.zero_grad()
        actorLoss.backward()
        self.actorOptimiser.step()

        self.criticOptimiser.zero_grad()
        criticLoss.backward()
        self.criticOptimiser.step()


    

# Carries out the steps of the A2C algorithm
class A2C:
    def __init__(self, inputDims, outputDims):
        # Hyperparameters set arbitrarily
        self.learningRate = 1e-4
        self.gamma = 0.99
        self.episodes = 1e-6

        # Probabilities stores the probability of taking a given action, 
        # rewards stores the reward of that action
        self.probabilities = []
        self.rewards = []

        self.network = ActorCritic(inputDims, outputDims)


    # Samples an action from the distribution according to mean
    #  and standard deviation of the estimated best action
    def chooseAction(self, state):
        means, stdDevs = self.network(state)

        # Defines a distibution, samples from it, then finds the 
        # probability of taking that action, for calculating loss
        distribution = Normal(means[0] + self.episodes, stdDevs[0] + self.episodes)
        action = distribution.sample()
        probability = distribution.log_prob(action)

        action = action.numpy()

        self.probabilities.append(probability)

        return action




# create a wrapper environment to save episode returns and episode lengths
envs_wrapper = gym.wrappers.RecordEpisodeStatistics(envs, deque_size=n_envs * n_updates)

critic_losses = []
actor_losses = []
entropies = []

# use tqdm to get a progress bar for training
for sample_phase in tqdm(range(n_updates)):
    # we don't have to reset the envs, they just continue playing
    # until the episode is over and then reset automatically

    # reset lists that collect experiences of an episode (sample phase)
    ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
    ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
    masks = torch.zeros(n_steps_per_update, n_envs, device=device)

    # at the start of training reset all envs to get an initial state
    if sample_phase == 0:
        states, info = envs_wrapper.reset(seed=42)

    # play n steps in our parallel environments to collect data
    for step in range(n_steps_per_update):
        # select an action A_{t} using S_{t} as input for the agent
        actions, action_log_probs, state_value_preds, entropy = agent.select_action(
            states
        )

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        states, rewards, terminated, truncated, infos = envs_wrapper.step(
            actions.cpu().numpy()
        )

        ep_value_preds[step] = torch.squeeze(state_value_preds)
        ep_rewards[step] = torch.tensor(rewards, device=device)
        ep_action_log_probs[step] = action_log_probs

        # add a mask (for the return calculation later);
        # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
        masks[step] = torch.tensor([not term for term in terminated])

    # calculate the losses for actor and critic
    critic_loss, actor_loss = agent.get_losses(
        ep_rewards,
        ep_action_log_probs,
        ep_value_preds,
        entropy,
        masks,
        gamma,
        lam,
        ent_coef,
        device,
    )

    # update the actor and critic networks
    agent.update_parameters(critic_loss, actor_loss)

    # log the losses and entropy
    critic_losses.append(critic_loss.detach().cpu().numpy())
    actor_losses.append(actor_loss.detach().cpu().numpy())
    entropies.append(entropy.detach().mean().cpu().numpy())


    """ plot the results """

# %matplotlib inline

rolling_length = 20
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
fig.suptitle(
    f"Training plots for {agent.__class__.__name__} in the LunarLander-v2 environment \n \
             (n_envs={n_envs}, n_steps_per_update={n_steps_per_update}, randomize_domain={randomize_domain})"
)

# episode return
axs[0][0].set_title("Episode Returns")
episode_returns_moving_average = (
    np.convolve(
        np.array(envs_wrapper.return_queue).flatten(),
        np.ones(rolling_length),
        mode="valid",
    )
    / rolling_length
)
axs[0][0].plot(
    np.arange(len(episode_returns_moving_average)) / n_envs,
    episode_returns_moving_average,
)
axs[0][0].set_xlabel("Number of episodes")

# entropy
axs[1][0].set_title("Entropy")
entropy_moving_average = (
    np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[1][0].plot(entropy_moving_average)
axs[1][0].set_xlabel("Number of updates")


# critic loss
axs[0][1].set_title("Critic Loss")
critic_losses_moving_average = (
    np.convolve(
        np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid"
    )
    / rolling_length
)
axs[0][1].plot(critic_losses_moving_average)
axs[0][1].set_xlabel("Number of updates")


# actor loss
axs[1][1].set_title("Actor Loss")
actor_losses_moving_average = (
    np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid")
    / rolling_length
)
axs[1][1].plot(actor_losses_moving_average)
axs[1][1].set_xlabel("Number of updates")

plt.tight_layout()
plt.show()

# def findNStepReturns(rewards, done, next_v_pred, gamma, n):
#     returns = torch.zeros_like(rewards)
#     futureReturn = next_v_pred
#     notDone = 1 - done
    
#     for timestep in reversed(range(n)):
#         returns[timestep] = futureReturn = rewards[timestep] + gamma * futureReturn * notDone[timestep]
        
#     return returns