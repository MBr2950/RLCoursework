import gymnasium as gym

env = gym.make('Humanoid-v4', render_mode="human")
observation, info = env.reset()

for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncuated, info = env.step(action)

    if terminated or truncuated:
        observation, info = env.reset()

env.close()