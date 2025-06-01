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
            print(reward_dict)
            reward = reward_dict['reward']
            group_rewards.append(reward) 
        rewards += group_rewards
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        means.append(mean_reward)
        stds.append(std_reward)
        maxs.append(np.max(group_rewards))
        mins.append(np.min(group_rewards))
        print(std_reward)
        print(repeated_ground_truths)
        for j in range(group_size):
            index = i * group_size + j 
            if normalize_by_std:
                advantages.append(group_rewards[j] - mean_reward)
                # advantages.append((group_rewards[j] - mean_reward) / (std_reward + advantage_eps))
            else:
                advantages.append(group_rewards[j] - mean_reward)
    return advantages, rewards, {'means': means, 'stds': stds, 'maxs': maxs, 'mins': mins}