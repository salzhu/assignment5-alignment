import torch 
import numpy as np 

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    reward_dict = reward_fn(rollout_responses, repeated_ground_truths)
    advantages = []
    means = []
    stds = []
    maxs = []
    mins = []
    rewards = []
    for i in range(len(rollout_responses) // group_size): 
        group_rewards = []
        for j in range(group_size):
            index = i * group_size + j 
            reward = reward_dict[index]['reward']
            group_rewards.append(reward) 
        rewards += group_rewards
        mean_reward = torch.mean(group_rewards)
        std_reward = torch.std(group_rewards)
        max_reward = torch.max(group_rewards) 
        min_reward = torch.min(min_reward)
        for j in range(group_size):
            index = i * group_size + j 
            if normalize_by_std:
                advantages.append((group_rewards[j] - mean_reward) / (std_reward + advantage_eps))
            else:
                advantages.append(group_rewards[j] - mean_reward)
    return advantages, rewards, {'means': means, 'stds': stds, 'maxs': maxs, 'mins': mins}