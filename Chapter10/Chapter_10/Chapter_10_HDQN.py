import math
import random
from collections import namedtuple, deque

import gymnasium as gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

import matplotlib.pyplot as plt

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer   = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Net, self).__init__()
        self.num_outputs = num_outputs
        
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state  = torch.FloatTensor(state).unsqueeze(0)
            action = self.forward(autograd.Variable(state)).max(1)[1]
            return action.item()
        else:
            return random.randrange(self.num_outputs)

env_id = "LunarLander-v3"
env = gym.make(env_id)

num_goals    = env.observation_space.shape[0]
num_actions  = env.action_space.n

model        = Net(2*num_goals, num_actions)
target_model = Net(2*num_goals, num_actions)

meta_model        = Net(num_goals, num_goals)
target_meta_model = Net(num_goals, num_goals)


optimizer      = optim.Adam(model.parameters())
meta_optimizer = optim.Adam(meta_model.parameters())

replay_buffer      = ReplayBuffer(10000)
meta_replay_buffer = ReplayBuffer(10000)

# NOTE: The original was copied from an example with 6 discrete states (np.zeros(6), oh[x - 1]). Here a goal is the
# index of one of LunarLander's num_goals (8) state values, so the one-hot vector needs num_goals entries; with 6
# the goal_state fed to `model` had 14 values instead of the 2*num_goals = 16 it expects.
def to_onehot(x):
    oh = np.zeros(num_goals)
    oh[x] = 1.
    return oh

def update(model, optimizer, replay_buffer, batch_size):
    if batch_size > len(replay_buffer):
        return
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    
    state      = autograd.Variable(torch.FloatTensor(state))
    next_state = autograd.Variable(torch.FloatTensor(next_state))
    action     = autograd.Variable(torch.LongTensor(action))
    reward     = autograd.Variable(torch.FloatTensor(reward))
    done       = autograd.Variable(torch.FloatTensor(done))
    
    q_value = model(state)
    q_value = q_value.gather(1, action.unsqueeze(1)).squeeze(1)
    
    next_q_value     = model(next_state).max(1)[0]
    expected_q_value = reward + 0.99 * next_q_value * (1 - done)
   
    loss = (q_value - autograd.Variable(expected_q_value.data)).pow(2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

num_frames = 100000
frame_idx  = 1

state, _ = env.reset()
done = False
all_rewards = []
episode_reward = 0

while frame_idx < num_frames:
    goal = meta_model.act(state, epsilon_by_frame(frame_idx))
    onehot_goal  = to_onehot(goal)
    
    meta_state = state
    extrinsic_reward = 0
    
    # NOTE: The original looped `while not done and goal != np.argmax(state)`. If the meta controller picked a goal
    # that was already satisfied, no step was taken, frame_idx never advanced and (once epsilon had decayed and the
    # greedy goal stopped changing) the program looped forever. Always take at least one step instead.
    while True:
        goal_state  = np.concatenate([state, onehot_goal])
        action = model.act(goal_state, epsilon_by_frame(frame_idx))
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        episode_reward   += reward
        extrinsic_reward += reward
        intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0

        replay_buffer.push(goal_state, action, intrinsic_reward, np.concatenate([next_state, onehot_goal]), done)
        state = next_state
        
        update(model, optimizer, replay_buffer, 32)
        update(meta_model, meta_optimizer, meta_replay_buffer, 32)
        frame_idx += 1
        
        if frame_idx % 1000 == 0 and len(all_rewards) > 0:
            n = 100 #mean reward of last 100 episodes
            print("frame %d, episodes %d, mean reward of last %d: %.1f" % (frame_idx, len(all_rewards), n, np.mean(all_rewards[-n:])))
            # The original used Jupyter's clear_output() and a blocking plt.show(); redraw one window without blocking.
            plt.figure(1) if plt.fignum_exists(1) else plt.figure(1, figsize=(20,5))
            plt.clf()
            plt.title(frame_idx)
            plt.plot([np.mean(all_rewards[i:i + n]) for i in range(0, len(all_rewards), n)])
            plt.pause(0.001)

        if done or goal == np.argmax(state):
            break

    meta_replay_buffer.push(meta_state, goal, extrinsic_reward, state, done)
        
    if done:
        state, _ = env.reset()
        done  = False
        all_rewards.append(episode_reward)
        episode_reward = 0