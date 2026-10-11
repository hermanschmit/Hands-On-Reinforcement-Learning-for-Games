"""
Minimal replacements for the parts of learn2learn and cherry used by the Chapter 14 meta-learning examples.

The examples were written against 2019 versions of those libraries. Their APIs have since changed (for example
learn2learn's TaskGenerator and gym environments are gone), and the current learn2learn release pins NumPy 1.x and
the old gym package, which would break the other chapters. Everything needed is small enough to write in plain
PyTorch, so it lives here instead.
"""

import math
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.func import functional_call


# ----------------------------------------------------------------------------------------------------------------
# Meta-learning wrappers (replacing learn2learn.algorithms.MAML / MetaSGD)

class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning. clone() returns a learner whose parameters are differentiable functions of the
    wrapped module's parameters; learner.adapt(loss) takes one gradient step, keeping the graph (second-order MAML)
    so the meta-loss can be backpropagated through the adaptation into the original parameters.
    """
    def __init__(self, module, lr, params=None):
        super(MAML, self).__init__()
        self.module = module
        self.lr = lr
        self.params = params  # None: use the module's own parameters

    def forward(self, *args, **kwargs):
        return self._call('forward', *args, **kwargs)

    def __getattr__(self, name):
        # Lets a learner call the wrapped module's other methods (e.g. policy.log_prob) with the adapted parameters.
        try:
            return super(MAML, self).__getattr__(name)
        except AttributeError:
            module = super(MAML, self).__getattr__('module')
            if callable(getattr(module, name, None)):
                return lambda *args, **kwargs: self._call(name, *args, **kwargs)
            raise

    def _call(self, method, *args, **kwargs):
        module = self.module
        if self.params is None:
            return getattr(module, method)(*args, **kwargs)
        if method == 'forward':
            return functional_call(module, self.params, args, kwargs)
        # functional_call only runs forward(), so temporarily point forward at the requested method.
        module.forward = getattr(module, method)
        try:
            return functional_call(module, self.params, args, kwargs)
        finally:
            del module.forward

    def _initial_params(self):
        if self.params is not None:
            return dict(self.params)
        return {name: p.clone() for name, p in self.module.named_parameters()}

    def _step_sizes(self, names):
        return [self.lr] * len(names)

    def clone(self):
        return type(self)(self.module, self.lr, self._initial_params())

    def adapt(self, loss):
        names, values = zip(*self.params.items())
        grads = torch.autograd.grad(loss, values, create_graph=True, allow_unused=True)
        self.params = {name: value if grad is None else value - lr * grad
                       for name, value, grad, lr in zip(names, values, grads, self._step_sizes(names))}


class MetaSGD(MAML):
    """Meta-SGD: like MAML, but every parameter has its own learnable step size, initialized to lr."""
    def __init__(self, module, lr, params=None, lrs=None):
        super(MetaSGD, self).__init__(module, lr, params)
        if lrs is None:
            lrs = nn.ParameterDict({name.replace('.', '_'): nn.Parameter(torch.full_like(p, lr))
                                    for name, p in module.named_parameters()})
        self.lrs = lrs

    def clone(self):
        return MetaSGD(self.module, self.lr, self._initial_params(), self.lrs)

    def _step_sizes(self, names):
        return [self.lrs[name.replace('.', '_')] for name in names]


def magic_box(x):
    """DiCE's MagicBox operator: evaluates to 1, but its gradient is the gradient of x."""
    return torch.exp(x - x.detach())


# ----------------------------------------------------------------------------------------------------------------
# Policy-gradient helpers (replacing cherry.td / cherry.pg / cherry.algorithms.a2c / cherry.normalize)

def discount(gamma, rewards, dones):
    """Discounted returns, restarting at the end of every episode."""
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.size(0))):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        returns[t] = running
    return returns


def generalized_advantage(gamma, tau, rewards, dones, values, next_value):
    next_values = torch.cat([values[1:], next_value.view(1, -1)], dim=0)
    td_errors = rewards + gamma * (1.0 - dones) * next_values - values
    return discount(gamma * tau, td_errors, dones)


def policy_loss(log_probs, advantages):
    return -(log_probs * advantages).mean()


def normalize(x, epsilon=1e-8):
    return (x - x.mean()) / (x.std() + epsilon)


class LinearValue(nn.Module):
    """Linear baseline on hand-crafted state/time features, fitted by regularized least squares."""
    def __init__(self, input_size, reg=1e-5):
        super(LinearValue, self).__init__()
        self.linear = nn.Linear(2 * input_size + 4, 1, bias=False)
        self.reg = reg

    def _features(self, states):
        length = states.size(0)
        ones = torch.ones(length, 1)
        al = torch.arange(length, dtype=torch.float32).view(-1, 1) / 100.0
        return torch.cat([states, states ** 2, al, al ** 2, al ** 3, ones], dim=1)

    def fit(self, states, returns):
        features = self._features(states.detach())
        A = features.t() @ features + self.reg * torch.eye(features.size(1))
        b = features.t() @ returns.detach()
        self.linear.weight.data = torch.linalg.lstsq(A, b).solution.t()

    def forward(self, states):
        return self.linear(self._features(states))


# ----------------------------------------------------------------------------------------------------------------
# Environment and episode collection (replacing learn2learn.gym's Particles2D-v1 and cherry.envs.Runner)

class Particles2DEnv(gym.Env):
    """
    A point in the plane has to move to a goal position. Each task has a different goal; the agent observes its
    position (not the goal), moves by up to 0.1 per step and is rewarded with minus the distance to the goal.
    """
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-0.1, 0.1, shape=(2,), dtype=np.float32)

    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.set_task(self.sample_tasks(1)[0])

    @property
    def state_size(self):
        return self.observation_space.shape[0]

    @property
    def action_size(self):
        return self.action_space.shape[0]

    def sample_tasks(self, num_tasks):
        goals = np.random.uniform(-0.5, 0.5, size=(num_tasks, 2))
        return [{'goal': goal} for goal in goals]

    def set_task(self, task):
        self._goal = task['goal']

    def reset(self, seed=None, options=None):
        super(Particles2DEnv, self).reset(seed=seed)
        self._state = np.zeros(2, dtype=np.float32)
        self._steps = 0
        return self._state.copy(), {}

    def step(self, action):
        self._state = (self._state + np.clip(action, -0.1, 0.1)).astype(np.float32)
        self._steps += 1
        x, y = self._state - self._goal
        reward = -math.sqrt(x ** 2 + y ** 2)
        terminated = bool(abs(x) < 0.01 and abs(y) < 0.01)
        truncated = self._steps >= self.max_steps
        return self._state.copy(), reward, terminated, truncated, {}


class Episodes(object):
    """Transitions from one or more episodes, as (T, size) tensors."""
    def __init__(self, states, actions, rewards, dones, next_states):
        self._data = dict(state=states, action=actions, reward=rewards, done=dones, next_state=next_states)

    def state(self):
        return self._data['state']

    def action(self):
        return self._data['action']

    def reward(self):
        return self._data['reward']

    def done(self):
        return self._data['done']

    def next_state(self):
        return self._data['next_state']


def run_episodes(env, policy, episodes):
    """Collect complete episodes with policy(state) -> action."""
    states, actions, rewards, dones, next_states = [], [], [], [], []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = policy(state_t)[0]
            next_state, reward, terminated, truncated, _ = env.step(action.numpy())
            done = terminated or truncated
            states.append(state_t[0])
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            next_states.append(torch.as_tensor(next_state, dtype=torch.float32))
            state = next_state
    return Episodes(torch.stack(states), torch.stack(actions), torch.tensor(rewards).view(-1, 1),
                    torch.tensor(dones).view(-1, 1), torch.stack(next_states))


# ----------------------------------------------------------------------------------------------------------------
# Few-shot classification tasks (replacing learn2learn.data.MetaDataset / TaskGenerator)

class TaskGenerator(object):
    """
    Samples N-way K-shot classification tasks from a labelled dataset. A task is a list of (input, label) pairs with
    `shots` examples from each of `ways` randomly chosen classes, relabelled 0..ways-1. Pass task=... to draw new
    examples of the same classes (e.g. a validation set for a training task).
    """
    def __init__(self, dataset, ways=2, shots=1):
        self.dataset = dataset
        self.ways = ways
        self.shots = shots
        targets = dataset.targets.tolist() if hasattr(dataset, 'targets') else [y for _, y in dataset]
        self.indices = {}
        for i, y in enumerate(targets):
            self.indices.setdefault(int(y), []).append(i)

    def sample(self, shots=None, task=None):
        shots = self.shots if shots is None else shots
        classes = random.sample(sorted(self.indices), self.ways) if task is None else task
        data = []
        for label, cls in enumerate(classes):
            for i in random.sample(self.indices[cls], shots):
                data.append((self.dataset[i][0], label))
        sampled = _Task(data)
        sampled.sampled_task = classes
        return sampled


class _Task(list):
    pass
