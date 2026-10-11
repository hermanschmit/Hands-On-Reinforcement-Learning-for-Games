import gymnasium as gym
# The Fetch robotics environments moved out of gym into the gymnasium-robotics package
# (pip install gymnasium-robotics, which also installs mujoco). They must be registered before gym.make.
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

env = gym.make('FetchReach-v4', render_mode='human')
env.reset()
for _ in range(1000):
    env.step(env.action_space.sample()) # take a random action
env.close()
