from os import system, name
import time
import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1')
env.reset()

def clear():
    if name == 'nt': 
        _ = system('cls')    
    else: 
        _ = system('clear') 

def act(V, env, gamma, policy, state, v):
    for action, action_prob in enumerate(policy[state]):                
        for state_prob, next_state, reward, end in env.P[state][action]:                                        
            v += action_prob * state_prob * (reward + gamma * V[next_state])                    
            V[state] = v
            
def eval_policy(policy, env, gamma=1.0, theta=1e-9, terms=1e9):     
    V = np.zeros(env.observation_space.n)  
    delta = 0
    for i in range(int(terms)): 
        for state in range(env.observation_space.n):            
            act(V, env, gamma, policy, state, v=0.0)         
        clear()
        print(V)
        time.sleep(1) 
        v = np.sum(V)
        if v - delta < theta:
            return V
        else:
            delta = v
    return V

policy = np.ones([env.unwrapped.observation_space.n, env.unwrapped.action_space.n]) / env.unwrapped.action_space.n
V = eval_policy(policy, env.unwrapped) 

print(policy, V)

