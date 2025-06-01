import torch 
import numpy as np 

"""
7.2 GRPO compute_group_normalized_rewards
- calculates raw rewards for each rollout response, 
- normalizes them within their groups, 
- and returns both the normalized and raw rewards along with metadata
"""
def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    
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
            reward_dict = reward_fn(rollout_responses[index], repeated_ground_truths[index])
            reward = reward_dict['reward']
            group_rewards.append(reward) 
        rewards += group_rewards
        mean_reward = torch.mean(torch.tensor(group_rewards))
        std_reward = torch.std(torch.tensor(group_rewards))
        means.append(mean_reward)
        stds.append(std_reward)
        maxs.append(np.max(group_rewards))
        mins.append(np.min(group_rewards))
        for j in range(group_size):
            index = i * group_size + j 
            if normalize_by_std:
                advantages.append((group_rewards[j] - mean_reward) / (std_reward + advantage_eps))
            else:
                advantages.append(group_rewards[j] - mean_reward)
    return advantages, rewards, {'means': means, 'stds': stds, 'maxs': maxs, 'mins': mins}

def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        ) -> torch.Tensor:
    return -1 * raw_rewards_or_advantages * policy_log_probs