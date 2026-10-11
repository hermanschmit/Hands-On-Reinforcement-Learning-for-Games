import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from common.layers import NoisyLinear
from common.replay_buffer import ReplayBuffer

#from torch.utils.tensorboard import SummaryWriter
# NOTE: The original used the `unityagents` package (ML-Agents 0.4, 2018), which no longer installs. It is ported here
# to its replacement, `mlagents_envs` (pip install mlagents-envs, which needs protobuf<=3.20: pip install
# protobuf==3.20.3). "Brains" were replaced by "behaviors", and stepping is now set_actions() + step() + get_steps().
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.registry import default_registry

# The desktop/GameAI.exe build in this repo can't be used: it was built with a 2019 version of ML-Agents that speaks
# an incompatible protocol, and its GameAI_Data folder was never checked in. Set unity_build to the path of your own
# build (made with ML-Agents Release 19, which matches mlagents-envs 0.28) to train on your game. Otherwise the
# script downloads Unity's prebuilt "Basic" example environment, a simple discrete-action game.
unity_build = None
#writer = SummaryWriter()

if unity_build is not None:
    env = UnityEnvironment(file_name=unity_build)
else:
    env = default_registry["Basic"].make()
env.reset()

# Get the default behavior (formerly the default brain)
behavior_name = list(env.behavior_specs)[0]

# The spec describes the observations and actions of the behavior we are controlling
spec = env.behavior_specs[behavior_name]

def env_reset():
    env.reset()
    decision_steps, _ = env.get_steps(behavior_name)
    return decision_steps.obs[0][0]

def env_step(action):
    """Gym-style step for a single agent: returns (next_state, reward, done)."""
    env.set_actions(behavior_name, ActionTuple(discrete=np.array([[action]], dtype=np.int32)))
    while True:
        env.step()
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(terminal_steps) > 0:
            return terminal_steps.obs[0][0], terminal_steps.reward[0], True
        if len(decision_steps) > 0:
            return decision_steps.obs[0][0], decision_steps.reward[0], False


class RainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()
        
        self.num_inputs   = num_inputs
        self.num_actions  = num_actions
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax
        
        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(32, 64)
        
        self.noisy_value1 = NoisyLinear(64, 64, use_cuda=False)
        self.noisy_value2 = NoisyLinear(64, self.num_atoms, use_cuda=False)
        
        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=False)
        self.noisy_advantage2 = NoisyLinear(64, self.num_atoms * self.num_actions, use_cuda=False)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)
        
        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)
        
        value     = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)
        
        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms), dim=1).view(-1, self.num_actions, self.num_atoms)
        
        return x
        
    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()
    
    def act(self, state):
        state = autograd.Variable(torch.FloatTensor(state).unsqueeze(0))
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action

num_atoms = 51
Vmin = -10
Vmax = 10

# Set the number of actions or action size (a single branch of discrete actions)
action_size = spec.action_spec.discrete_branches[0]

# Set the size of state observations or state size
state_size = spec.observation_specs[0].shape[0]

current_model = RainbowDQN(state_size, action_size, num_atoms, Vmin, Vmax)
target_model  = RainbowDQN(state_size, action_size, num_atoms, Vmin, Vmax)

optimizer = optim.Adam(current_model.parameters(), 0.001)

replay_buffer = ReplayBuffer(10000)

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())
    
update_target(current_model, target_model)

def projection_distribution(next_state, rewards, dones):
    batch_size  = next_state.size(0)
    
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)
    support = torch.linspace(Vmin, Vmax, num_atoms)
    
    # NOTE: The original multiplied next_dist by the support here and then used those products (some negative) as
    # the probabilities that get projected below, so the target was not a probability distribution. The product is
    # only needed to pick the greedy next action.
    next_dist   = target_model(next_state).data.cpu()
    next_action = (next_dist * support).sum(2).max(1)[1]
    next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
    next_dist   = next_dist.gather(1, next_action).squeeze(1)
        
    rewards = rewards.unsqueeze(1).expand_as(next_dist)
    dones   = dones.unsqueeze(1).expand_as(next_dist)
    support = support.unsqueeze(0).expand_as(next_dist)
    
    Tz = rewards + (1 - dones) * 0.99 * support
    Tz = Tz.clamp(min=Vmin, max=Vmax)
    b  = (Tz - Vmin) / delta_z
    l  = b.floor().long()
    u  = b.ceil().long()
    # When b lands exactly on an atom, l == u and both weights below are 0, which would drop that probability mass.
    l[(u > 0) & (l == u)] -= 1
    u[(l < (num_atoms - 1)) & (l == u)] += 1
        
    offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, num_atoms)

    proj_dist = torch.zeros(next_dist.size())    
    proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
    proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
        
    return proj_dist

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size) 

    state      = autograd.Variable(torch.FloatTensor(np.float32(state)))
    next_state = autograd.Variable(torch.FloatTensor(np.float32(next_state)))
    action     = autograd.Variable(torch.LongTensor(action))
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(np.float32(done))

    proj_dist = projection_distribution(next_state, reward, done)
    
    dist = current_model(state)
    action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
    dist = dist.gather(1, action).squeeze(1)
    dist.data.clamp_(0.01, 0.99)
    loss = -(autograd.Variable(proj_dist) * dist.log()).sum(1)
    loss  = loss.mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    current_model.reset_noise()
    target_model.reset_noise()
    
    return loss

def plot(iteration, rewards, losses, ep_reward, cmodel, tmodel, input): 
    print("Outputing Iteration " + str(iteration))
    #writer.add_scalar('Train/Rewards', rewards[-1], iteration)
    #writer.add_scalar('Train/Losses', losses[-1], iteration)     
    #writer.add_scalar('Train/Episode', ep_reward, iteration)  
    #ac = cmodel.act(input)
    #at = tmodel.act(input)
    #writer.add_scalar("Train/Advantage", ac-at, iteration)    
    #writer.flush()

iterations = 150000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

# get initial state of the unity environment
state = env_reset()

for iteration in range(1, iterations + 1):
    action = current_model.act(state)

    next_state, reward, done = env_step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if done:
        state = env_reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
        
    if iteration % 1000 == 0 and len(all_rewards) > 0:
        print("Iteration %d, episodes %d, mean reward of last 10: %.2f" % (iteration, len(all_rewards), np.mean(all_rewards[-10:])))

    if iteration % 20 == 0 and len(all_rewards) > 0 and len(losses) > 0:
        plot(iteration, all_rewards, losses, episode_reward, current_model, target_model, state)
        
    if iteration % 1000 == 0:
        update_target(current_model, target_model)

# Shut down the Unity process
env.close()
