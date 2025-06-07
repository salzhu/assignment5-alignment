import torch 
import numpy as np 
from typing import Literal
from cs336_alignment.sft_helper import masked_normalize

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

"""
7.2 GRPO compute_grpo_clip_loss
- computes the per-token GRPO-Clip loss
"""
def compute_grpo_clip_loss(
        advantages: torch.Tensor,
        policy_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        cliprange: float,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    policy_ratio = torch.div(torch.exp(policy_log_probs), torch.exp(old_log_probs))
    clipped_policy_ratio = torch.clamp(policy_ratio, min=1-cliprange, max=1+cliprange)

    loss = -1 * torch.minimum(advantages * policy_ratio, advantages * clipped_policy_ratio)
    clipped = advantages * clipped_policy_ratio < advantages * policy_ratio
    return loss, {'clipped': clipped}

"""
7.2 GRPO compute_policy_gradient_loss
- convenience wrapper that dispatches to the correct loss routine 
    - (no_baseline, reinforce_with_baseline, or grpo_clip)
- returns both the per-token loss and any auxiliary statistics
"""
def compute_policy_gradient_loss(
        policy_log_probs: torch.Tensor,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == 'no_baseline': 
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {'loss': loss}
    if loss_type == 'reinforce_with_baseline': 
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {'loss': loss}
    if loss_type == 'grpo_clip':
        loss, metadata = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        metadata['loss'] = loss
        return loss, metadata

"""
7.2 GRPO masked_mean
- averages tensor elements while respecting a boolean mask
"""
def masked_mean(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None = None,
        ) -> torch.Tensor:
    tensor = tensor * mask 
    mean_tensor = torch.sum(tensor, dim=dim) / torch.sum(mask, dim=dim)
    return mean_tensor 

"""
7.2 GRPO grpo_microbatch_train_step
- implements a single micro-batch update for GRPO, including policy-gradient loss, 
  averaging with a mask, and gradient scaling
"""
def grpo_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        raw_rewards: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        old_log_probs: torch.Tensor | None = None,
        cliprange: float | None = None,
        length_normalize=False,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    print(advantages)
    print(advantages.shape, raw_rewards.shape, policy_log_probs.shape, response_mask.shape, old_log_probs.shape)
    loss, metadata = compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, 
                                        old_log_probs, cliprange)
    if length_normalize:
        max_gen_len = response_mask.shape[-1]
        masked_normalize_loss = masked_normalize(
            loss, response_mask, normalize_constant=max_gen_len) / gradient_accumulation_steps
        masked_normalize_loss.backward()
        metadata['loss_len_normalized'] = masked_normalize_loss
        metadata['policy_log_probs_grad'] = policy_log_probs.grad
        return masked_normalize_loss, metadata
        
    masked_mean_loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    masked_mean_loss.backward()
    metadata['loss_masked'] = masked_mean_loss
    metadata['policy_log_probs_grad'] = policy_log_probs.grad
    return 0, 0 #masked_mean_loss, metadata