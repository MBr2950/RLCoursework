import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

rewards1 = {}
with open('DDPG1.json', 'r') as file:
    rewards1 = json.load(file)

rewards1a = {}
with open('DDPG2.json', 'r') as file:
    rewards1a = json.load(file)

rewards1.update(rewards1a)

rewards1b = {}
with open('DDPG3.json', 'r') as file:
    rewards1b = json.load(file)

rewards1.update(rewards1b)

rewards1c = {}
with open('DDPG4.json', 'r') as file:
    rewards1c = json.load(file)

rewards1.update(rewards1c)

rewards1d = {}
with open('DDPG5.json', 'r') as file:
    rewards1d = json.load(file)

rewards1.update(rewards1d)

rewards1e = {}
with open('DDPG6.json', 'r') as file:
    rewards1e = json.load(file)

rewards1.update(rewards1e)

rewards1f = {}
with open('DDPG7.json', 'r') as file:
    rewards1f = json.load(file)

rewards1.update(rewards1f)

rewards1g = {}
with open('DDPG8.json', 'r') as file:
    rewards1g = json.load(file)

rewards1.update(rewards1g)

rewards1h = {}
with open('DDPG9.json', 'r') as file:
    rewards1h = json.load(file)

rewards1.update(rewards1h)

rewards1i = {}
with open('DDPG10.json', 'r') as file:
    rewards1i = json.load(file)

rewards1.update(rewards1i)

#print(min(rewards1c['Run 4'])) # 2 5? 7 9

# Create a DataFrame for Algorithm 1
df_green = pd.DataFrame(rewards1)

# Display min, max, average, and rolling across iterations for Algorithm 1
rolling_min_green = df_green.min(axis=1)
rolling_max_green = df_green.max(axis=1)
rolling_avg_green = df_green.mean(axis=1).rolling(window=20, min_periods=1).mean()

# Apply a moving average to min and max values for smoothing
smoothed_min_green = rolling_min_green.rolling(window=100, min_periods=1).mean()
smoothed_max_green = rolling_max_green.rolling(window=100, min_periods=1).mean()

peak_index_green = rolling_avg_green.idxmax()

# Plotting the rewards data with smoothed min, max, average, and rolling lines for Algorithm 1 (green)
plt.figure(figsize=(10, 6))
plt.plot(smoothed_min_green.index, smoothed_min_green, color='green', alpha=0.2, linewidth=0.5)
plt.plot(smoothed_max_green.index, smoothed_max_green, color='green', alpha=0.2, linewidth=0.5)
plt.plot(rolling_avg_green.index, rolling_avg_green, label='DDPG', color='green')
plt.fill_between(smoothed_min_green.index, smoothed_min_green, smoothed_max_green, color='green', alpha=0.1)

# Add vertical lines at each episode
plt.xlim(0, len(rewards1['Run 1']))
plt.axhline(0, color='black', linewidth=0.8)
for episode in range(0, len(rewards1['Run 1']), 100):  # Adjust the step size as needed
    plt.axvline(episode, color='gray', linestyle=':', linewidth=0.8)

plt.axhline(rolling_avg_green.loc[peak_index_green], color='red', linestyle='--', linewidth=2)

#plt.ylim(-5000,1001)

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()
