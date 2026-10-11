import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from multiprocessing_env import SubprocVecEnv
from minipacman import MiniPacman
from environment_model import EnvModel
from actor_critic import OnPolicy, ActorCritic, RolloutStorage

import matplotlib.pyplot as plt
from tqdm import tqdm

#USE_CUDA = torch.cuda.is_available()
#Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

# NOTE: The first two colors were swapped relative to Chapter_14_Imagination.py, which trains the environment model
# (env_model_regular) with this mapping, so imagined frames were decoded with the wrong colors.
pixels = (
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0), 
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0), 
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0)
)
pixel_to_onehot = {pix:i for i, pix in enumerate(pixels)} 
num_pixels = len(pixels)

task_rewards = {
    "regular": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "avoid":   [0.1, -0.1, -5, -10, -20],
    "hunt":    [0, 1, 10, -20],
    "ambush":  [0, -0.1, 10, -20],
    "rush":    [0, -0.1, 9.9]
}
reward_to_onehot = {mode: {reward:i for i, reward in enumerate(task_rewards[mode])} for mode in task_rewards.keys()}

# Each pixel channel is 0 or 1, so (r, g, b) -> 4r + 2g + b indexes a lookup table of class numbers.
# (Vectorized: the original looped over every pixel in Python, which dominated the run time.)
_code_to_class = np.full(8, -1, dtype=np.int64)
for _i, (_r, _g, _b) in enumerate(pixels):
    _code_to_class[int(4 * _r + 2 * _g + _b)] = _i

def pix_to_target(next_states):
    pix = np.round(next_states.transpose(0, 2, 3, 1).reshape(-1, 3)).astype(np.int64)
    return _code_to_class[4 * pix[:, 0] + 2 * pix[:, 1] + pix[:, 2]]

def target_to_pix(imagined_states):
    return np.array(pixels)[imagined_states]

def rewards_to_target(mode, rewards):
    target = []
    for reward in rewards:
        target.append(reward_to_onehot[mode][reward])
    return target
    
def displayImage(image, step, reward):
    s = str(step) + " " + str(reward)
    plt.title(s)
    plt.imshow(image)
    plt.show()

mode = "regular"
num_envs = 16

def make_env():
    def _thunk():
        env = MiniPacman(mode, 1000)
        return env

    return _thunk

#envs = [make_env() for i in range(num_envs)]
#envs = SubprocVecEnv(envs)

#state_shape = envs.observation_space.shape
#num_actions = envs.action_space.n
#num_rewards = len(task_rewards[mode])

class RolloutEncoder(nn.Module):
    def __init__(self, in_shape, num_rewards, hidden_size):
        super(RolloutEncoder, self).__init__()
        
        self.in_shape = in_shape
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        self.gru = nn.GRU(self.feature_size() + num_rewards, hidden_size)
        
    def forward(self, state, reward):
        num_steps  = state.size(0)
        batch_size = state.size(1)
        
        state = state.view(-1, *self.in_shape)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)
    
    
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)

class I2A(OnPolicy):
    def __init__(self, in_shape, num_actions, num_rewards, hidden_size, imagination, full_rollout=True):
        super(I2A, self).__init__()
        
        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards
        
        self.imagination = imagination
        
        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        
        self.encoder = RolloutEncoder(in_shape, num_rewards, hidden_size)
        
        if full_rollout:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + num_actions * hidden_size, 256),
                nn.ReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + hidden_size, 256),
                nn.ReLU(),
            )
        
        self.critic  = nn.Linear(256, 1)
        self.actor   = nn.Linear(256, num_actions)
        
    def forward(self, state):
        batch_size = state.size(0)
        
        imagined_state, imagined_reward = self.imagination(state.data)
        hidden = self.encoder(autograd.Variable(imagined_state), autograd.Variable(imagined_reward))
        hidden = hidden.view(batch_size, -1)
        
        state = self.features(state)
        state = state.view(state.size(0), -1)
        
        x = torch.cat([state, hidden], 1)
        x = self.fc(x)
        
        logit = self.actor(x)
        value = self.critic(x)
        
        return logit, value
        
    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)


class ImaginationCore(object):
    def __init__(self, num_rolouts, in_shape, num_actions, num_rewards, env_model, distil_policy, full_rollout=True):
        self.num_rolouts  = num_rolouts
        self.in_shape      = in_shape
        self.num_actions   = num_actions
        self.num_rewards   = num_rewards
        self.env_model     = env_model
        self.distil_policy = distil_policy
        self.full_rollout  = full_rollout
        
    def __call__(self, state):
        state      = state.cpu()
        batch_size = state.size(0)

        rollout_states  = []
        rollout_rewards = []

        if self.full_rollout:
            # NOTE: The original repeated along a new first dimension (unsqueeze(0).repeat(num_actions, ...)), which orders
            # the copies action-major, while the actions below and I2A's hidden.view(batch_size, -1) assume batch-major
            # order. That paired states with the wrong actions and mixed different agents' imagined rollouts together.
            state = state.unsqueeze(1).repeat(1, self.num_actions, 1, 1, 1).view(-1, *self.in_shape)
            action = torch.LongTensor([[i] for i in range(self.num_actions)]*batch_size)
            action = action.view(-1)
            rollout_batch_size = batch_size * self.num_actions
        else:
            action = self.distil_policy.act(autograd.Variable(state))
            action = action.data.cpu()
            rollout_batch_size = batch_size

        for step in range(self.num_rolouts):
            onehot_action = torch.zeros(rollout_batch_size, self.num_actions, *self.in_shape[1:])
            onehot_action[range(rollout_batch_size), action] = 1
            inputs = torch.cat([state, onehot_action], 1)

            imagined_state, imagined_reward = self.env_model(autograd.Variable(inputs))

            imagined_state  = F.softmax(imagined_state, dim=1).max(1)[1].data.cpu()
            imagined_reward = F.softmax(imagined_reward, dim=1).max(1)[1].data.cpu()

            imagined_state = target_to_pix(imagined_state.numpy())
            imagined_state = torch.FloatTensor(imagined_state).view(rollout_batch_size, *self.in_shape)

            onehot_reward = torch.zeros(rollout_batch_size, self.num_rewards)
            onehot_reward[range(rollout_batch_size), imagined_reward] = 1

            rollout_states.append(imagined_state.unsqueeze(0))
            rollout_rewards.append(onehot_reward.unsqueeze(0))

            state  = imagined_state
            action = self.distil_policy.act(autograd.Variable(state))
            action = action.data.cpu()
        
        return torch.cat(rollout_states), torch.cat(rollout_rewards)

def main():

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    state_shape = envs.observation_space.shape
    num_actions = envs.action_space.n
    num_rewards = len(task_rewards[mode])

    full_rollout = True

    env_model     = EnvModel(envs.observation_space.shape, num_pixels, num_rewards)
    env_model.load_state_dict(torch.load("env_model_" + mode))

    distil_policy = ActorCritic(envs.observation_space.shape, envs.action_space.n)
    distil_optimizer = optim.Adam(distil_policy.parameters())

    imagination = ImaginationCore(1, state_shape, num_actions, num_rewards, env_model, distil_policy, full_rollout=full_rollout)

    actor_critic = I2A(state_shape, num_actions, num_rewards, 256, imagination, full_rollout=full_rollout)
    #rmsprop hyperparams:
    lr    = 7e-4
    eps   = 1e-5
    alpha = 0.99
    optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    #if USE_CUDA:
    #    env_model     = env_model.cuda()
    #    distil_policy = distil_policy.cuda()
    #    actor_critic  = actor_critic.cuda()

    gamma = 0.99
    entropy_coef = 0.01
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    num_steps = 5
    num_frames = int(10e5)

    rollout = RolloutStorage(num_steps, num_envs, envs.observation_space.shape)
    #rollout.cuda()

    all_rewards = []
    all_losses  = []

    state = envs.reset()
    current_state = torch.FloatTensor(np.float32(state))

    rollout.states[0].copy_(current_state)

    episode_rewards = torch.zeros(num_envs, 1)
    final_rewards   = torch.zeros(num_envs, 1)

    for i_update in tqdm(range(num_frames)):

        for step in range(num_steps):
            #if USE_CUDA:
            #    current_state = current_state.cuda()
            action = actor_critic.act(autograd.Variable(current_state))

            next_state, reward, done, _ = envs.step(action.squeeze(1).cpu().data.numpy())

            reward = torch.FloatTensor(reward).unsqueeze(1)
            episode_rewards += reward
            masks = torch.FloatTensor(1-np.array(done)).unsqueeze(1)
            final_rewards *= masks
            final_rewards += (1-masks) * episode_rewards
            episode_rewards *= masks

            #if USE_CUDA:
            #    masks = masks.cuda()

            current_state = torch.FloatTensor(np.float32(next_state))
            rollout.insert(step, current_state, action.data, reward, masks)


        _, next_value = actor_critic(autograd.Variable(rollout.states[-1]))
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, gamma)

        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
            autograd.Variable(rollout.states[:-1]).view(-1, *state_shape),
            autograd.Variable(rollout.actions).view(-1, 1)
        )
    
        distil_logit, _, _, _ = distil_policy.evaluate_actions(
            autograd.Variable(rollout.states[:-1]).view(-1, *state_shape),
            autograd.Variable(rollout.actions).view(-1, 1)
        )
        
        # NOTE: The original was missing the minus sign, so it maximized the cross-entropy and pushed the distilled
        # policy away from the I2A policy instead of towards it.
        distil_loss = -0.01 * (F.softmax(logit, dim=1).detach() * F.log_softmax(distil_logit, dim=1)).sum(1).mean()

        values = values.view(num_steps, num_envs, 1)
        action_log_probs = action_log_probs.view(num_steps, num_envs, 1)
        advantages = autograd.Variable(returns) - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(autograd.Variable(advantages.data) * action_log_probs).mean()

        optimizer.zero_grad()
        loss = value_loss * value_loss_coef + action_loss - entropy * entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), max_grad_norm)
        optimizer.step()
    
        # NOTE: The original called optimizer.step() here, so the distilled policy was never trained and the I2A agent
        # took a second step with the same gradients.
        distil_optimizer.zero_grad()
        distil_loss.backward()
        distil_optimizer.step()
    
        if i_update % 100 == 0:
            all_rewards.append(final_rewards.mean().item())
            all_losses.append(loss.item())

            # Redraw one window without blocking (the original blocked training in plt.show()).
            plt.figure(1) if plt.fignum_exists(1) else plt.figure(1, figsize=(20,5))
            plt.clf()
            plt.subplot(131)
            plt.title('epoch %s. reward: %s' % (i_update, np.mean(all_rewards[-10:])))
            plt.plot(all_rewards)
            plt.subplot(132)
            plt.title('loss %s' % all_losses[-1])
            plt.plot(all_losses)
            plt.pause(0.001)
        
        rollout.after_update()

    torch.save(actor_critic.state_dict(), "i2a_" + mode)

if __name__ == '__main__':
    main()


