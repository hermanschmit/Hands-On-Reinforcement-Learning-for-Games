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
            
def evaluate(V, action_values, env, gamma, state):
    for action in range(env.action_space.n):
        for prob, next_state, reward, terminated in env.P[state][action]:
            action_values[action] += prob * (reward + gamma * V[next_state])
    return action_values

def lookahead(env, state, V, gamma):
    action_values = np.zeros(env.action_space.n)
    return evaluate(V, action_values, env, gamma, state)

def improve_policy(env, gamma=1.0, terms=1e9):    
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    evals = 1
    for i in range(int(terms)):
        stable = True       
        V = eval_policy(policy, env, gamma=gamma)
        for state in range(env.observation_space.n):
            current_action = np.argmax(policy[state])
            action_value = lookahead(env, state, V, gamma)
            best_action = np.argmax(action_value)
            if current_action != best_action:
                stable = False                
                policy[state] = np.eye(env.action_space.n)[best_action]
            evals += 1                
            if stable:
                return policy, V

def eval_policy(policy, env, gamma=1.0, theta=1e-5, terms=1e9):     
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

def value_iteration(env, gamma=1.0, theta=1e-9, terms=1e9):
    V = np.zeros(env.observation_space.n)
    for i in range(int(terms)):
        delta = 0
        for state in range(env.observation_space.n):
            action_value = lookahead(env, state, V, gamma)
            best_action_value = np.max(action_value)
            delta = max(delta, np.abs(V[state] - best_action_value))
            V[state] = best_action_value             
        if delta < theta: break
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for state in range(env.observation_space.n):
        action_value = lookahead(env, state, V, gamma)
        best_action = np.argmax(action_value)
        policy[state, best_action] = 1.0
    return policy, V

def play(env, episodes, policy):
    wins = 0
    total_reward = 0
    for episode in range(episodes):
        term = False
        state, _ = env.reset()
        while not term:
            action = np.argmax(policy[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            term = terminated or truncated
            total_reward += reward
            state = next_state
            if term and reward == 1.0:
                wins += 1
    average_reward = total_reward / episodes
    return wins, total_reward, average_reward

policy, V = improve_policy(env.unwrapped)
print(policy, V)

wins, total, avg = play(env, 1000, policy)
print(wins)

policy, V = value_iteration(env.unwrapped)
print(policy, V)

wins, total, avg = play(env, 1000, policy)
print(wins)


