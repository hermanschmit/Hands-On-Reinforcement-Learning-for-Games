
# NOTE: This example originally used keras-rl, which has not been maintained since 2018 and only runs on
# TensorFlow 1.x with the old gym API (its keras-rl2 fork stops at TensorFlow 2.15 / NumPy 1.x). It is ported here to
# Stable-Baselines3 (pip install stable-baselines3), the maintained library in the same spirit: a complete DQN agent,
# configured and trained in a few lines.
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import DQN


ENV_NAME = 'CartPole-v1'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
nb_actions = env.action_space.n

# Next, we build a simple model: two hidden layers of 256 relu units, with nb_actions linear outputs. (The keras-rl
# version used three layers of 16 units; with Stable-Baselines3's epsilon-greedy DQN that network is unstable and
# usually collapses to ~10 reward, while this one reliably reaches the maximum score of 500.)
policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU)

# Finally, we configure our agent. buffer_size plays the role of keras-rl's SequentialMemory limit and
# learning_starts its nb_steps_warmup. These are the tuned CartPole settings from the RL Baselines3 Zoo; the
# keras-rl ones (soft target updates, a Boltzmann policy) do not carry over to Stable-Baselines3's DQN.
dqn = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, learning_rate=2.3e-3, batch_size=64, buffer_size=100000,
          learning_starts=1000, gamma=0.99, target_update_interval=10, train_freq=256, gradient_steps=128,
          exploration_fraction=0.16, exploration_final_eps=0.04, seed=123, verbose=1)
print(dqn.policy)

# Okay, now it's time to learn something! Training is not visualized (Stable-Baselines3 has no equivalent of
# keras-rl's visualize=True, which slowed training down a lot anyway). You can always safely abort the training
# prematurely using Ctrl + C.
dqn.learn(total_timesteps=50000, log_interval=20)

# After training is done, we save the final weights.
dqn.save('dqn_{}_weights'.format(ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.
test_env = gym.make(ENV_NAME, render_mode='human')
for episode in range(5):
    state, _ = test_env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = dqn.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
        episode_reward += reward
    print('Episode {}: reward: {}'.format(episode + 1, episode_reward))
test_env.close()
