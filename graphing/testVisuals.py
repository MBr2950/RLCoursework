import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming rewards is a list containing rewards data for each episode in each iteration
# For demonstration, let's assume 5 iterations and 1000 episodes per iteration
num_iterations = 5
num_episodes = 1000

# Create a dictionary to store rewards data for each iteration for Algorithm 1 (green)
rewards_data_green = {}

# Loop through iterations for Algorithm 1
for seed in range(1, num_iterations + 1):
    rewards = np.random.normal(0, 0.3, size=1000) * seed
    rewards_data_green[f'Iteration {seed}'] = rewards

# Create a DataFrame for Algorithm 1
df_green = pd.DataFrame(rewards_data_green)

# Create a dictionary to store rewards data for each iteration for Algorithm 2 (red)
rewards_data_red = {}

# Loop through iterations for Algorithm 2
for seed in range(1, num_iterations + 1):
    rewards = np.random.normal(0, 0.3, size=1000) * seed
    rewards += 4
    rewards_data_red[f'Iteration {seed}'] = rewards

# Create a DataFrame for Algorithm 2
df_red = pd.DataFrame(rewards_data_red)

# Create a dictionary to store rewards data for each iteration for Algorithm 3 (yellow)
rewards_data_yellow = {}

# Loop through iterations for Algorithm 3
for seed in range(1, num_iterations + 1):
    rewards = np.random.normal(0, 0.3, size=1000) * seed
    rewards += 8
    rewards_data_yellow[f'Iteration {seed}'] = rewards

# Create a DataFrame for Algorithm 3
df_yellow = pd.DataFrame(rewards_data_yellow)

# Display min, max, average, and rolling across iterations for Algorithm 1
min_values_green = df_green.min(axis=1)
max_values_green = df_green.max(axis=1)
avg_values_green = df_green.mean(axis=1)
rolling_min_green = min_values_green.rolling(window=5, min_periods=1).min()
rolling_max_green = max_values_green.rolling(window=5, min_periods=1).max()
rolling_avg_green = avg_values_green.rolling(window=10, min_periods=1).mean()

# Display min, max, average, and rolling across iterations for Algorithm 2
min_values_red = df_red.min(axis=1)
max_values_red = df_red.max(axis=1)
avg_values_red = df_red.mean(axis=1)
rolling_min_red = min_values_red.rolling(window=5, min_periods=1).min()
rolling_max_red = max_values_red.rolling(window=5, min_periods=1).max()
rolling_avg_red = avg_values_red.rolling(window=10, min_periods=1).mean()

# Display min, max, average, and rolling across iterations for Algorithm 3
min_values_yellow = df_yellow.min(axis=1)
max_values_yellow = df_yellow.max(axis=1)
avg_values_yellow = df_yellow.mean(axis=1)
rolling_min_yellow = min_values_yellow.rolling(window=5, min_periods=1).min()
rolling_max_yellow = max_values_yellow.rolling(window=5, min_periods=1).max()
rolling_avg_yellow = avg_values_yellow.rolling(window=10, min_periods=1).mean()

# Plotting the rewards data with min, max, average, and rolling lines for Algorithm 1 (green)
plt.figure(figsize=(10, 6))
plt.plot(rolling_min_green.index, rolling_min_green, color='green', alpha=0.2, linewidth=0.5)
plt.plot(rolling_max_green.index, rolling_max_green, color='green', alpha=0.2, linewidth=0.5)
plt.plot(rolling_avg_green.index, rolling_avg_green, label='Algorithm 1 (Green)', color='green')
plt.fill_between(rolling_min_green.index, rolling_min_green, rolling_max_green, color='green', alpha=0.1)

# Plotting the rewards data with min, max, average, and rolling lines for Algorithm 2 (red)
plt.plot(rolling_min_red.index, rolling_min_red, color='blue', alpha=0.2, linewidth=0.5)
plt.plot(rolling_max_red.index, rolling_max_red, color='blue', alpha=0.2, linewidth=0.5)
plt.plot(rolling_avg_red.index, rolling_avg_red, label='Algorithm 2 (Blue)', color='blue')
plt.fill_between(rolling_min_red.index, rolling_min_red, rolling_max_red, color='blue', alpha=0.1)

# Plotting the rewards data with min, max, average, and rolling lines for Algorithm 3 (yellow)
plt.plot(rolling_min_yellow.index, rolling_min_yellow, color='gray', alpha=0.2, linewidth=0.5)
plt.plot(rolling_max_yellow.index, rolling_max_yellow, color='gray', alpha=0.2, linewidth=0.5)
plt.plot(rolling_avg_yellow.index, rolling_avg_yellow, label='Algorithm 3 (Grey)', color='gray')
plt.fill_between(rolling_min_yellow.index, rolling_min_yellow, rolling_max_yellow, color='gray', alpha=0.1)

# Add vertical lines at each episode
plt.xlim(0, num_episodes)
plt.axhline(0, color='black', linewidth=0.8)
for episode in range(0, num_episodes, 100):  # Adjust the step size as needed
    plt.axvline(episode, color='gray', linestyle=':', linewidth=0.8)

plt.title('Rewards Over Episodes for Each Iteration with Rolling Metrics')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()
