import torch
import numpy as np
import functools

from config import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cat(list_of_tensors, dim=0):
    """
    Concatenate a list of tensors.
    """
    return functools.reduce(lambda x, y: torch.cat([x, y], dim=dim), list_of_tensors)


def catcat(list_of_lists_of_tensors, dim_outer=0, dim_inner=0):
    """
    Recursively concatenate a list of tensors.
    """
    return cat([cat(inner_list, dim_inner) for inner_list in list_of_lists_of_tensors], dim_outer)


def discounted(vals, gamma=0.99):
    """
    Computes the discounted sum as used for the return in RL.
    """
    G = 0
    discounted = torch.zeros_like(vals)
    for i in np.arange(vals.shape[-1]-1, -1, -1):
        G = vals[..., i] + gamma * G
        discounted[..., i] = G

    return discounted


def advantage(rewards, values):
    """
    Computes the advantage of the given returns as compared to the estimated values, optionally using GAE.
    """
    if GAE_LAMBDA == 0:
        returns = discounted(rewards, GAMMA)
        advantage = returns - values
    else:
        # Generalized Advantage Estimation (GAE) https://arxiv.org/abs/1506.02438
        # via https://github.com/inoryy/reaver/blob/master/reaver/agents/base/actor_critic.py
        values = torch.cat([values, torch.zeros((values.shape[0], 1, 1)).to(DEVICE)], dim=2)
        deltas = rewards + GAMMA * values[..., 1:] - values[..., :-1]
        advantage = discounted(deltas, GAMMA * GAE_LAMBDA)

    return advantage


class Buffer:

    """
    Utility class to gather a replay buffer. Computes returns and advantages over logged trajectories.
    """

    def __init__(self):
        self.count = 0  # number of trajectories
        # environment
        self.sources = []
        self.targets = []
        self.rewards = []
        self.values = []
        # expert related
        self.expert_actions = []
        # student related
        self.actions = []
        self.action_logits = []
        self.action_logprobs = []

    def __len__(self):
        return self.count

    def start_trajectory(self):
        """
        Initializes the list into which all samples of a trajectory are gathered.
        """
        #
        self.count += 1
        self.sources += [[]]
        self.targets += [[]]
        self.rewards += [[]]
        self.values += [[]]
        self.expert_actions += [[]]
        self.actions += [[]]
        self.action_logits += [[]]
        self.action_logprobs += [[]]

    def log_step(self, observation, state_value, reward, expert_action,
                 action, action_logit, action_logprob
                 ):
        """
        Logs a single step in a trajectory.
        """
        self.sources[-1].append(observation[0].detach())
        self.targets[-1].append(observation[1].detach())
        self.expert_actions[-1].append(expert_action.detach())
        self.rewards[-1].append(reward.detach())
        self.values[-1].append(state_value.detach())

        self.actions[-1].append(action.detach())
        self.action_logits[-1].append(torch.cat([a[:, None, :] for a in action_logit], dim=1).detach())
        self.action_logprobs[-1].append(action_logprob.detach())

    def get_returns_and_advantages(self):
        """
        Computes the return and advantage per trajectory in the buffer.
        """
        returns = [discounted(cat(rewards, dim=-1), GAMMA).transpose(2, 1)
                   for rewards in self.rewards]  # per trajectory
        advantages = [advantage(cat(rewards, dim=-1), cat(values, dim=-1)).transpose(2, 1)
                      for rewards, values in zip(self.rewards, self.values)]
        return returns, advantages

    def get_samples(self):
        """
        Gather all samples in the buffer for use in a torch.utils.data.TensorDataset.
        """
        samples = [self.sources, self.targets,
                   self.expert_actions, self.values,
                   self.actions, self.action_logits, self.action_logprobs]
        samples += self.get_returns_and_advantages()
        return [catcat(sample) for sample in samples]

    def clear(self):
        """
        Clears the buffer and all its trajectory lists.
        """
        self.count = 0
        self.sources.clear()
        self.targets.clear()
        self.rewards.clear()
        self.expert_actions.clear()
        self.values.clear()
        self.actions.clear()
        self.action_logits.clear()
        self.action_logprobs.clear()
