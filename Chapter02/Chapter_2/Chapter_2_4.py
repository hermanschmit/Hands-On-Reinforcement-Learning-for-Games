from os import system, name
import time
import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', render_mode='ansi')
env.reset()

def clear():
    if name == 'nt': 
        _ = system('cls')    
    else: 
        _ = system('clear') 

for _ in range(1000):
    clear()
    print(env.render())
    time.sleep(.5)
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample()) # take a random action
    if terminated or truncated:
        env.reset()
env.close()
