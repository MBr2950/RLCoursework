import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

rewards1 = {}
with open('A2C1steps.json', 'r') as file:
    rewards1 = json.load(file)

rewards1a = {}
with open('A2C2steps.json', 'r') as file:
    rewards1a = json.load(file)

rewards1.update(rewards1a)

rewards1b = {}
with open('A2C3steps.json', 'r') as file:
    rewards1b = json.load(file)

rewards1.update(rewards1b)

rewards1c = {}
with open('A2C4steps.json', 'r') as file:
    rewards1c = json.load(file)

rewards1.update(rewards1c)

rewards1d = {}
with open('A2C5steps.json', 'r') as file:
    rewards1d = json.load(file)

rewards1.update(rewards1d)

rewards1e = {}
with open('A2C6steps.json', 'r') as file:
    rewards1e = json.load(file)

rewards1.update(rewards1e)

rewards1f = {}
with open('A2C7steps.json', 'r') as file:
    rewards1f = json.load(file)

rewards1.update(rewards1f)

rewards1g = {}
with open('A2C8steps.json', 'r') as file:
    rewards1g = json.load(file)

rewards1.update(rewards1g)

rewards1h = {}
with open('A2C9steps.json', 'r') as file:
    rewards1h = json.load(file)

rewards1.update(rewards1h)

rewards1i = {}
with open('A2C10steps.json', 'r') as file:
    rewards1i = json.load(file)

rewards1.update(rewards1i)

rewards2 = {}
with open('A2C11steps.json', 'r') as file:
    rewards2 = json.load(file)

rewards2a = {}
with open('A2C12steps.json', 'r') as file:
    rewards2a = json.load(file)

rewards2.update(rewards2a)

rewards2b = {}
with open('A2C13steps.json', 'r') as file:
    rewards2b = json.load(file)

rewards2.update(rewards2b)

rewards2c = {}
with open('A2C14steps.json', 'r') as file:
    rewards2c = json.load(file)

rewards2.update(rewards2c)

rewards2d = {}
with open('A2C15steps.json', 'r') as file:
    rewards2d = json.load(file)

rewards2.update(rewards2d)

rewards2e = {}
with open('A2C16steps.json', 'r') as file:
    rewards2e = json.load(file)

rewards2.update(rewards2e)

rewards2f = {}
with open('A2C17steps.json', 'r') as file:
    rewards2f = json.load(file)

rewards2.update(rewards2f)

rewards2g = {}
with open('A2C18steps.json', 'r') as file:
    rewards2g = json.load(file)

rewards2.update(rewards2g)

rewards2h = {}
with open('A2C19steps.json', 'r') as file:
    rewards2h = json.load(file)

rewards2.update(rewards2h)

rewards2i = {}
with open('A2C20steps.json', 'r') as file:
    rewards2i = json.load(file)

rewards2.update(rewards2i)

rewards3 = {}
with open('A2C21steps.json', 'r') as file:
    rewards3 = json.load(file)

rewards3a = {}
with open('A2C22steps.json', 'r') as file:
    rewards3a = json.load(file)

rewards3.update(rewards3a)

rewards3b = {}
with open('A2C23steps.json', 'r') as file:
    rewards3b = json.load(file)

rewards3.update(rewards3b)

rewards3c = {}
with open('A2C24steps.json', 'r') as file:
    rewards3c = json.load(file)

rewards3.update(rewards3c)

rewards3d = {}
with open('A2C25steps.json', 'r') as file:
    rewards3d = json.load(file)

rewards3.update(rewards3d)

rewards3e = {}
with open('A2C26steps.json', 'r') as file:
    rewards3e = json.load(file)

rewards3.update(rewards3e)

rewards3f = {}
with open('A2C27steps.json', 'r') as file:
    rewards3f = json.load(file)

rewards3.update(rewards3f)

rewards3g = {}
with open('A2C28steps.json', 'r') as file:
    rewards3g = json.load(file)

rewards3.update(rewards3g)

rewards3h = {}
with open('A2C29steps.json', 'r') as file:
    rewards3h = json.load(file)

rewards3.update(rewards3h)

rewards3i = {}
with open('A2C30steps.json', 'r') as file:
    rewards3i = json.load(file)

rewards3.update(rewards3i)

rewards4 = {}
with open('RANDOM1steps.json', 'r') as file:
    rewards4 = json.load(file)

rewards4a = {}
with open('RANDOM2steps.json', 'r') as file:
    rewards4a = json.load(file)

rewards4.update(rewards4a)

rewards4b = {}
with open('RANDOM3steps.json', 'r') as file:
    rewards4b = json.load(file)

rewards4.update(rewards4b)

rewards4c = {}
with open('RANDOM4steps.json', 'r') as file:
    rewards4c = json.load(file)

rewards4.update(rewards4c)

rewards4d = {}
with open('RANDOM5steps.json', 'r') as file:
    rewards4d = json.load(file)

rewards4.update(rewards4d)

rewards4e = {}
with open('RANDOM6steps.json', 'r') as file:
    rewards4e = json.load(file)

rewards4.update(rewards4e)

rewards4f = {}
with open('RANDOM7steps.json', 'r') as file:
    rewards4f = json.load(file)

rewards4.update(rewards4f)

rewards4g = {}
with open('RANDOM8steps.json', 'r') as file:
    rewards4g = json.load(file)

rewards4.update(rewards4g)

rewards4h = {}
with open('RANDOM9steps.json', 'r') as file:
    rewards4h = json.load(file)

rewards4.update(rewards4h)

rewards4i = {}
with open('RANDOM10steps.json', 'r') as file:
    rewards4i = json.load(file)

rewards4.update(rewards4i)

# Create a DataFrame for Algorithms
df_green = pd.DataFrame(rewards1)
df_blue = pd.DataFrame(rewards2)
df_gold = pd.DataFrame(rewards3)
df_grey = pd.DataFrame(rewards4)

# Display min, max, average, and rolling across iterations for Algorithms
rolling_min_green = df_green.min(axis=1)
rolling_max_green = df_green.max(axis=1)
rolling_avg_green = df_green.mean(axis=1).rolling(window=100, min_periods=1).mean()

rolling_min_blue = df_blue.min(axis=1)
rolling_max_blue = df_blue.max(axis=1)
rolling_avg_blue = df_blue.mean(axis=1).rolling(window=100, min_periods=1).mean()

rolling_min_gold = df_gold.min(axis=1)
rolling_max_gold = df_gold.max(axis=1)
rolling_avg_gold = df_gold.mean(axis=1).rolling(window=100, min_periods=1).mean()

rolling_min_grey = df_grey.min(axis=1)
rolling_max_grey = df_grey.max(axis=1)
rolling_avg_grey = df_grey.mean(axis=1).rolling(window=100, min_periods=1).mean()

# Apply a moving average to min and max values for smoothing
smoothed_min_green = rolling_min_green.rolling(window=100, min_periods=1).mean()
smoothed_max_green = rolling_max_green.rolling(window=100, min_periods=1).mean()

peak_index_green = rolling_avg_green.idxmax()

smoothed_min_blue = rolling_min_blue.rolling(window=100, min_periods=1).mean()
smoothed_max_blue = rolling_max_blue.rolling(window=100, min_periods=1).mean()

peak_index_blue = rolling_avg_blue.idxmax()

smoothed_min_gold = rolling_min_gold.rolling(window=100, min_periods=1).mean()
smoothed_max_gold = rolling_max_gold.rolling(window=100, min_periods=1).mean()

peak_index_gold = rolling_avg_gold.idxmax()

smoothed_min_grey = rolling_min_grey.rolling(window=100, min_periods=1).mean()
smoothed_max_grey = rolling_max_grey.rolling(window=100, min_periods=1).mean()

peak_index_grey = rolling_avg_grey.idxmax()

# Plotting the rewards data with smoothed min, max, average, and rolling lines for Algorithm 1 (green)
plt.figure(figsize=(10, 6))
plt.plot(smoothed_min_green.index, smoothed_min_green, color='green', alpha=0.2, linewidth=0.5)
plt.plot(smoothed_max_green.index, smoothed_max_green, color='green', alpha=0.2, linewidth=0.5)
plt.plot(rolling_avg_green.index, rolling_avg_green, label='A2C (TD Advantage) (27-64-64-32-32)', color='green')
plt.fill_between(smoothed_min_green.index, smoothed_min_green, smoothed_max_green, color='green', alpha=0.1)

plt.plot(smoothed_min_blue.index, smoothed_min_blue, color='blue', alpha=0.2, linewidth=0.5)
plt.plot(smoothed_max_blue.index, smoothed_max_blue, color='blue', alpha=0.2, linewidth=0.5)
plt.plot(rolling_avg_blue.index, rolling_avg_blue, label='A2C (GAE) (27-64-64-32-32)', color='blue')
plt.fill_between(smoothed_min_blue.index, smoothed_min_blue, smoothed_max_blue, color='blue', alpha=0.1)

#plt.plot(smoothed_min_gold.index, smoothed_min_gold, color='purple', alpha=0.2, linewidth=0.5)
#plt.plot(smoothed_max_gold.index, smoothed_max_gold, color='purple', alpha=0.2, linewidth=0.5)
#plt.plot(rolling_avg_gold.index, rolling_avg_gold, label='A2C3 (27-256-64-32)', color='purple')
#plt.fill_between(smoothed_min_gold.index, smoothed_min_gold, smoothed_max_gold, color='purple', alpha=0.1)

plt.plot(smoothed_min_grey.index, smoothed_min_grey, color='black', alpha=0.2, linewidth=0.5)
plt.plot(smoothed_max_grey.index, smoothed_max_grey, color='black', alpha=0.2, linewidth=0.5)
plt.plot(rolling_avg_grey.index, rolling_avg_grey, label='Random Agent', color='black')
plt.fill_between(smoothed_min_grey.index, smoothed_min_grey, smoothed_max_grey, color='black', alpha=0.1)

# Add vertical lines at each episode
plt.xlim(0, len(rewards1['Run 1']))
plt.axhline(0, color='black', linewidth=0.8)
for episode in range(0, len(rewards1['Run 1']), 100):  # Adjust the step size as needed
    plt.axvline(episode, color='gray', linestyle=':', linewidth=0.8)

plt.axhline(rolling_avg_blue.loc[peak_index_blue], color='red', linestyle='--', linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Steps')
plt.legend()
plt.show()
