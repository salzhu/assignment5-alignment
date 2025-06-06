import torch 
import wandb
import random 
import numpy as np
import json 
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from mock import patch
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import argparse
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from cs336_alignment.sft_helper import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import evaluate_vllm

prompt_path = '/home/c-salzhu/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt'
val_path = '/data/a5-alignment/MATH/validation.jsonl'
replacement = "{question}"

sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024
)
sampling_params.stop = ["</answer>"]
sampling_params.include_stop_str_in_output = True

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.2):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
        )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

def train_sft(model_name, train_path, n_examples, n_eval,
              grad_accum_steps, learning_rate, batch_size, eval_steps, filter_correct=False):
    
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2",).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []
    outputs = []

    with open(train_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if filter_correct:
                rewards = r1_zero_reward_fn(data['response'], data['ground_truth'])
                if rewards['answer_reward'] == 1:
                    prompts.append(data['prompt'])
                    outputs.append(data["response"])
            else:
                prompts.append(data['prompt'])
                outputs.append(data["response"])
    print(len(prompts))
    tokenized_dict = tokenize_prompt_and_output(prompts, outputs, tokenizer)

    input_ids_tensor = torch.tensor(tokenized_dict['input_ids'][:n_examples])
    label_ids_tensor = torch.tensor(tokenized_dict['labels'][:n_examples])
    mask_tensor = torch.tensor(tokenized_dict['response_mask'][:n_examples])

    dataset = TensorDataset(input_ids_tensor, label_ids_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size//grad_accum_steps, shuffle=True)

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

    llm = init_vllm(model_name, 'cuda', 0)
    load_policy_into_vllm_instance(model, llm)

    wandb.init(
        project="a5-sft",
        name=f"sft_n{n_examples}_lr{learning_rate}_bs{batch_size}",  # Set your run name here
    )

    # Setup wandb metrics
    wandb.define_metric("train_step") # the x‑axis for training
    wandb.define_metric("eval_step") # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    end = 0
    for idx, (input, labels, mask) in enumerate(dataloader):
        input = input.to('cuda')
        labels = labels.to('cuda')
        mask = mask.to('cuda')
        policy_log_probs = get_response_log_probs(model, input, labels, False)['log_probs']
        loss, metadata = sft_microbatch_train_step(policy_log_probs, mask, grad_accum_steps, normalize_constant=1.0)

        # wandb log the train loss 
        wandb.log({
            "train/train_loss": loss.item(),
            "train_step": idx + 1
        })

        if (idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update weights every `grad_accum_steps` batches.
            optimizer.step()

            # Zero gradients every `grad_accum_steps` batches.
            optimizer.zero_grad()

        if (idx + 1) % eval_steps == 0: 
            load_policy_into_vllm_instance(model, llm)
            indices = random.sample(range(len(eval_prompts)), n_eval) 
            eval_prompts_small = [eval_prompts[i] for i in indices]
            eval_answers_small = [eval_answers[i] for i in indices]
            eval_full_dataset_small = [eval_full_dataset[i] for i in indices]
            evals = evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts_small, eval_answers_small, eval_full_dataset_small, 
                                  sampling_params, 'temp.json')
            correct = 0
            for i in range(len(evals)):
                if evals[i]['rewards']['answer_reward'] == 1: 
                    correct += 1

            log = {'eval/accuracy': correct / len(evals),'eval_step': (idx + 1) // eval_steps}
            wandb.log(log)
            end = (idx + 1) // eval_steps

    load_policy_into_vllm_instance(model, llm)
    evals = evaluate_vllm(llm, r1_zero_reward_fn, eval_prompts, eval_answers, eval_full_dataset, sampling_params, 'temp.json')

    correct = 0
    for i in range(len(evals)):
        if evals[i]['rewards']['answer_reward'] == 1: 
            correct += 1

    log = {'eval/accuracy': correct / len(evals),'eval_step': end + 2}
    wandb.log(log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/data/a5-alignment/models/Qwen2.5-Math-1.5B')
    parser.add_argument('--train_path', type=str, default='/data/a5-alignment/MATH/sft.jsonl')
    parser.add_argument('--n_examples', type=int, default=128)
    parser.add_argument('--n_eval', type=int, default=512)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_steps', type=int, default=8)
    parser.add_argument('--filter_correct', type=bool, default=False)

    args = parser.parse_args()

    train_sft(args.model_name, args.train_path, args.n_examples, args.n_eval,
              args.grad_accum_steps, args.learning_rate, args.batch_size, args.eval_steps, args.filter_correct)