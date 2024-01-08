# Reinforcement Learning - CM30359

## Introduction

This repository is dedicated to the exploration of various reinforcement learning algorithms, specifically applied to the Ant-v4 MuJoCo environment through OpenAI Gym. The focus is on understanding and implementing different strategies to analyze their effectiveness in a simulated environment.

## How to run the code

- The `Report Graphs` folder contains all algorithms with seeding that were used in the final report graphs
- Within each algorithm folder, there is a program called "Import visualiser v3.py" that generates graphs from .json files
- There are further folders of different variations of each algorithm that contain the 10 separate programs with seeding used
- Running each program will generate two separate json files that contains the logged rewards and steps of that run

## Repository Contents

- `Random Agent`: A baseline from which to compare the other algorithms.

- `DDPG Algorithm`: Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm.

- `Reinforce Algorithm`: Contains the implementation of the Reinforce algorithm.

- `Reinforce A2C Algorithm`: This section implements the Reinforce Algorithm with the Advantage Actor-Critic (A2C) approach.

- `Proximal Policy Optimisation (PPO) Algorithm`: This section implements the Proximal Policy Optimisation (PPO) algorithm.
