import torch 
import wandb
import json 
from vllm import LLM, SamplingParams
from mock import patch
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import argparse
import random 
import copy
from typing import Literal
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from cs336_alignment.sft_experiment import init_vllm, load_policy_into_vllm_instance
from cs336_alignment.sft_helper import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.grpo_helper import compute_group_normalized_rewards, grpo_microbatch_train_step
from cs336_alignment.math_baseline import evaluate_vllm

prompt_path = '/home/c-salzhu/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt'
val_path = '/data/a5-alignment/MATH/validation.jsonl'
train_path = '/data/a5-alignment/MATH/train.jsonl'
replacement = "{question}"

def train_grpo(model_name, 
               n_grpo_steps: int = 200,
               learning_rate: float = 1e-5,
               advantage_eps: float = 1e-6,
               rollout_batch_size: int = 256,
               group_size: int = 8,
               sampling_temperature: float = 1.0,
               sampling_min_tokens: int = 4, # As in Expiter, disallow empty string responses,
               sampling_max_tokens: int = 1024,
               epochs_per_rollout_batch: int = 1, # On-policy
               train_batch_size: int = 256, # On-policy,
               gradient_accumulation_steps: int = 128, # microbatch size is 2, will fit on H100,
               gpu_memory_utilization: float = 0.2,
               loss_type: Literal[
                   "no_baseline",
                   "reinforce_with_baseline",
                   "grpo_clip",] = "reinforce_with_baseline",
                use_std_normalization: bool = True,
                cliprange = 0.2,
                eval_steps=256,
                length_normalize=False,
                n_eval=1024
            ):
    policy = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2",).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_prompts = []
    train_answers = []
    train_full_dataset = []
    with open(prompt_path, "r") as file:
        prompt = file.read()
    with open(train_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            problem = data["problem"]
            train_prompts.append(prompt.replace(replacement, problem))
            train_answers.append(data["answer"])
            train_full_dataset.append(data)

    eval_prompts = []
    eval_answers = []
    eval_full_dataset = []
    with open(prompt_path, "r") as file:
        prompt = file.read()
    with open(val_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            problem = data["problem"]
            eval_prompts.append(prompt.replace(replacement, problem))
            eval_answers.append(data["answer"])
            eval_full_dataset.append(data)

    llm = init_vllm(model_name, 'cuda', 0, gpu_memory_utilization=gpu_memory_utilization)
    old_llm = init_vllm(model_name, 'cuda', 0)
    load_policy_into_vllm_instance(policy, llm)
    load_policy_into_vllm_instance(policy, old_llm)

    train_sampling_params = SamplingParams(
        temperature=sampling_temperature, top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        n=group_size,
        seed=0,
    )
    train_sampling_params.stop = ["</answer>"]
    train_sampling_params.include_stop_str_in_output = True

    eval_sampling_params = SamplingParams(
        temperature=sampling_temperature, top_p=1.0,
        max_tokens=sampling_max_tokens,
        min_tokens=sampling_min_tokens,
        seed=0,
    )
    eval_sampling_params.stop = ["</answer>"]
    eval_sampling_params.include_stop_str_in_output = True

    optimizer = torch.optim.AdamW(
                    policy.parameters(),
                    lr=learning_rate,
                    weight_decay=0.0,
                    betas=(0.9, 0.95),
                )
    
    wandb.init(
        project="a5-grpo",
        name=f"grpo_lr{learning_rate}_{loss_type}_lennorm{length_normalize}",  # Set your run name here
    )

    # Setup wandb metrics
    wandb.define_metric("train_step") # the x‑axis for training
    wandb.define_metric("eval_step") # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    assert rollout_batch_size % group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = rollout_batch_size // group_size

    assert train_batch_size % gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps

    assert train_batch_size >= group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )

    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    train_step = 0

    for _ in range(n_grpo_steps):
        
        # sample questions
        train_indices = random.sample(range(len(train_prompts)), n_prompts_per_rollout_batch) 
        old_policy = copy.deepcopy(policy)
        load_policy_into_vllm_instance(old_policy, old_llm)
        # get dataset 
        train_prompts_small = [train_prompts[i] for i in train_indices]
        train_answers_small = [train_answers[i] for i in train_indices]

        outputs = old_llm.generate(train_prompts_small, train_sampling_params)

        rollout_responses = []
        repeated_ground_truths = []
        prompts = []
        for i in range(len(train_prompts_small)):
            for j in range(len(outputs[i].outputs)):
                rollout_responses.append(outputs[i].outputs[j].text)
            repeated_ground_truths += group_size * [train_answers_small[i]]
            prompts += group_size * [train_prompts_small[i]]

        advantages, rew_rewards, _ = compute_group_normalized_rewards(r1_zero_reward_fn, rollout_responses, 
                                                      repeated_ground_truths, group_size, 
                                                      advantage_eps, use_std_normalization)
        advantages = torch.stack(advantages)
        
        tokenized_dict = tokenize_prompt_and_output(prompts, rollout_responses, tokenizer)

        input_ids_tensor = torch.tensor(tokenized_dict['input_ids'])
        label_ids_tensor = torch.tensor(tokenized_dict['labels'])
        mask_tensor = torch.tensor(tokenized_dict['response_mask'])

        old_policy.to('cuda')
        old_policy_log_probs = []
        for i in range(len(input_ids_tensor)):
            old_policy_log_probs.append(get_response_log_probs(old_policy, 
                                                               torch.unsqueeze(input_ids_tensor[i],0).to('cuda'), 
                                                               torch.unsqueeze(label_ids_tensor[i],0).to('cuda'), 
                                                               False)['log_probs'].detach())
        old_policy.to('cpu')
        old_policy_log_probs = torch.stack(old_policy_log_probs).to('cuda')
        torch.cuda.empty_cache()

        dataset = TensorDataset(input_ids_tensor, label_ids_tensor, mask_tensor, old_policy_log_probs, advantages)
        dataloader = DataLoader(dataset, batch_size=micro_train_batch_size, shuffle=True)
        
        for epoch in range(epochs_per_rollout_batch):

            for idx, (input, labels, mask, old_log_probs, advantage) in enumerate(dataloader):
                input = input.to('cuda')
                labels = labels.to('cuda')
                mask = mask.to('cuda')
                policy_log_probs = get_response_log_probs(policy, input, labels, True)
                token_entropy = policy_log_probs['token_entropy']
                policy_log_probs = policy_log_probs['log_probs']
                advantage = advantage.to('cuda')
                advantage = torch.unsqueeze(advantage,-1)
                loss, metadata = grpo_microbatch_train_step(
                    policy_log_probs, mask, gradient_accumulation_steps, loss_type, 
                    advantage, advantage, old_log_probs, cliprange,length_normalize
                )

                wandb.log({
                    "train/train_loss": loss.item(),
                    "train/token_entropy": torch.mean(token_entropy).item(),
                    "train_step": train_step+1
                })

                if (train_step+1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    # Update weights every `grad_accum_steps` batches.
                    optimizer.step()

                    # Zero gradients every `grad_accum_steps` batches.
                    optimizer.zero_grad()
                
                if (train_step + 1) % eval_steps == 0: 
                    load_policy_into_vllm_instance(policy, llm)
                    indices = random.sample(range(len(eval_prompts)), n_eval) 
                    eval_prompts_small = [eval_prompts[i] for i in indices]
                    eval_answers_small = [eval_answers[i] for i in indices]
                    eval_full_dataset_small = [eval_full_dataset[i] for i in indices]
                    evals = evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts_small, eval_answers_small, 
                                          eval_full_dataset_small, 
                                          eval_sampling_params, 'temp.json')
                    correct = 0
                    rewards = 0
                    for i in range(len(evals)):
                        if evals[i]['rewards']['answer_reward'] == 1: 
                            correct += 1
                        rewards += evals[i]['rewards']['reward']

                    log = {'eval/accuracy': correct / len(evals),'eval_step': (train_step + 1) // eval_steps}
                    wandb.log(log)
                    log = {'eval/rewards': rewards,'eval_step': (train_step + 1) // eval_steps}
                    wandb.log(log)
                    end = (train_step + 1) // eval_steps
                
                train_step += 1
            torch.cuda.empty_cache()

    load_policy_into_vllm_instance(policy, llm)
    evals = evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts, eval_answers, eval_full_dataset, 
                            eval_sampling_params, 'temp.json')

    correct = 0
    rewards = 0
    for i in range(len(evals)):
        if evals[i]['rewards']['answer_reward'] == 1: 
            correct += 1
        rewards += evals[i]['rewards']['reward']

    log = {'eval/accuracy': correct / len(evals),'eval_step': end + 2}
    wandb.log(log)
    log = {'eval/rewards': rewards,'eval_step': (train_step + 1) // eval_steps}
    wandb.log(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/data/a5-alignment/models/Qwen2.5-Math-1.5B')
    parser.add_argument('--train_path', type=str, default='/data/a5-alignment/MATH/sft.jsonl')

    parser.add_argument('--n_grpo_steps', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--advantage_eps', type=float, default=1e-6)
    parser.add_argument('--rollout_batch_size', type=int, default=256)
    parser.add_argument('--group_size', type=int, default=8)

    parser.add_argument('--sampling_temperature', type=float, default=1.0)
    parser.add_argument('--sampling_min_tokens', type=int, default=4)
    parser.add_argument('--sampling_max_tokens', type=int, default=1024)

    parser.add_argument('--epochs_per_rollout_batch', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=128)
    
    parser.add_argument('--loss_type', type=str, default='reinforce_with_baseline')

    parser.add_argument('--gpu_memory_utilization', type=float, default=0.2)
    parser.add_argument('--use_std_normalization', type=int, default=0)
    parser.add_argument('--len_normalize', type=int, default=0)
    parser.add_argument('--cliprange', type=float, default=0.2)

    parser.add_argument('--eval_steps', type=int, default=256)
    parser.add_argument('--n_eval', type=int, default=1024)

    args = parser.parse_args()

    use_length_normalization = False 
    if args.use_length_normalization == 1:
        use_length_normalization = True 

    use_std_normalization = False 
    if args.use_std_normalization == 1:
        use_std_normalization = True 

    train_grpo(args.model_name, 
               args.n_grpo_steps, 
               args.learning_rate, 
               args.advantage_eps, 
               args.rollout_batch_size, 
               args.group_size, 
               args.sampling_temperature, 
               args.sampling_min_tokens, 
               args.sampling_max_tokens, 
               args.epochs_per_rollout_batch,
               args.train_batch_size,
               args.gradient_accumulation_steps,
               args.gpu_memory_utilization,
               args.loss_type,
               use_std_normalization,
               args.cliprange,
               args.eval_steps,
               use_length_normalization,
               n_eval=1024
            )
