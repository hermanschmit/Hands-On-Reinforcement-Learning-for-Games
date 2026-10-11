import os
import sys
import time
# simple_rl (last released in 2019) still calls time.clock(), which was removed in Python 3.8.
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

from simple_rl.agents import QLearningAgent, RandomAgent, RMaxAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.run_experiments import run_agents_on_mdp

# Setup MDP.
mdp = GridWorldMDP(width=4, height=3, init_loc=(1, 1), goal_locs=[(4, 3)], lava_locs=[(4, 2)], gamma=0.95, walls=[(2, 2)], slip_prob=0.05)

# Setup Agents.
ql_agent = QLearningAgent(actions=mdp.get_actions())
rmax_agent = RMaxAgent(actions=mdp.get_actions())
rand_agent = RandomAgent(actions=mdp.get_actions())

# Run experiment and make plot.
# simple_rl opens the plot with the macOS-only `open` command, so on other systems open the saved PDF ourselves.
run_agents_on_mdp([ql_agent, rmax_agent, rand_agent], mdp, instances=5, episodes=50, steps=10,
                  open_plot=(sys.platform == 'darwin'))
if sys.platform == 'win32':
    os.startfile(os.path.abspath(os.path.join('results', str(mdp), 'cumulative_reward.pdf')))
