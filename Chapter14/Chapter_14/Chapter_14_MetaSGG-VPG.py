#!/usr/bin/env python3

"""
Trains a 2-layer MLP with MetaSGD-VPG.
"""

import random

import numpy as np
import torch as th
from torch import optim
from tqdm import tqdm

# NOTE: The original used learn2learn (MAML/MetaSGD, the Particles2D environment) and cherry (episode collection,
# baseline, advantages). Their current releases no longer support this code, so minimal PyTorch versions of the
# same pieces are provided in meta_learning.py.
from meta_learning import MetaSGD, LinearValue, Particles2DEnv, discount, generalized_advantage, magic_box, \
    normalize, policy_loss, run_episodes
from policies import DiagNormalPolicy


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
    returns = discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = th.zeros(1, device=values.device)
    return generalized_advantage(tau=tau,
                                 gamma=gamma,
                                 rewards=rewards,
                                 dones=dones,
                                 values=bootstraps,
                                 next_value=next_value)


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    # Update policy and baseline
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = normalize(advantages).detach()
    return policy_loss(log_probs, advantages)


def main(
        experiment='dev',
        env_name='Particles2D-v1',
        adapt_lr=0.1,
        meta_lr=0.01,
        adapt_steps=1,
        num_iterations=200,
        meta_bsz=20,
        adapt_bsz=20,
        tau=1.00,
        gamma=0.99,
        seed=42,
):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

    # NOTE: The original vectorized the environment with learn2learn's AsyncVectorEnv; a single environment is
    # plenty for this tiny task.
    env = Particles2DEnv()
    policy = DiagNormalPolicy(env.state_size, env.action_size)
    # NOTE: The original passed meta_lr as the adaptation step size, leaving adapt_lr unused.
    # The optimizer also has to include meta_learner.parameters(), which hold Meta-SGD's learned per-parameter
    # step sizes; the original only optimized policy.parameters(), so the step sizes never changed.
    meta_learner = MetaSGD(policy, lr=adapt_lr)
    # NOTE: The original passed env.action_size as LinearValue's second argument, which is its regularization
    # coefficient, not a size.
    baseline = LinearValue(env.state_size)
    opt = optim.Adam(meta_learner.parameters(), lr=meta_lr)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_loss = 0.0
        iteration_reward = 0.0
        for task_config in tqdm(env.sample_tasks(meta_bsz)):  # Samples a new config
            learner = meta_learner.clone()
            env.set_task(task_config)
            env.reset()

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = run_episodes(env, learner, episodes=adapt_bsz)
                loss = maml_a2c_loss(train_episodes, learner, baseline, gamma, tau)
                learner.adapt(loss)

            # Compute Validation Loss
            valid_episodes = run_episodes(env, learner, episodes=adapt_bsz)
            loss = maml_a2c_loss(valid_episodes, learner, baseline, gamma, tau)
            iteration_loss += loss
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz

        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / meta_bsz
        print('adaptation_reward', adaptation_reward)
        all_rewards.append(adaptation_reward)

        adaptation_loss = iteration_loss / meta_bsz
        print('adaptation_loss', adaptation_loss.item())

        opt.zero_grad()
        adaptation_loss.backward()
        opt.step()


if __name__ == '__main__':
    main()
