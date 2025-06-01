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

"""
7.2 GRPO compute_naive_policy_gradient_loss
- computes the per-token policy-gradient loss using raw rewards or pre-computed advantages
"""
def compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        ) -> torch.Tensor:
    return -1 * raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_ratio = policy_log_probs / old_log_probs
    clipped_policy_ratio = torch.clamp(policy_ratio, min=1 - cliprange, max=1 + cliprange)
    return -1 * min(advantages * policy_ratio, advantages * clipped_policy_ratio)