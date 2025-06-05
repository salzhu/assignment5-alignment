import torch 
import wandb
import json 
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from mock import patch
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
import argparse

from cs336_alignment.sft_helper import tokenize_prompt_and_output, get_response_log_probs, sft_microbatch_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.math_baseline import evaluate_vllm

prompt_path = 'prompts/r1_zero.prompt'
val_path = '/data/a5-alignment/MATH/validation.json'
replacement = "{question}"

sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
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

def train_sft(model_name, train_path, n_examples, 
              grad_accum_steps, learning_rate, batch_size, eval_steps, filter_correct=False):
    
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2",).to('cuda:0')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompts = []
    outputs = []

    with open(train_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data['prompt'])
            outputs.append(data["response"])
    tokenized_dict = tokenize_prompt_and_output(prompts, outputs, tokenizer)

    input_ids_tensor = torch.tensor(tokenized_dict['input_ids'][:n_examples])
    label_ids_tensor = torch.tensor(tokenized_dict['labels'][:n_examples])
    mask_tensor = torch.tensor(tokenized_dict['response_mask'][:n_examples])

    dataset = TensorDataset(input_ids_tensor, label_ids_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size//grad_accum_steps, shuffle=True)

    prompts = []
    answers = []
    full_dataset = []
    with open(prompt_path, "r") as file:
        prompt = file.read()
    with open(val_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            problem = data["problem"]
            prompts.append(prompt.replace(replacement, problem))
            answers.append(data["answer"])
            full_dataset.append(data)

    llm = init_vllm(model_name, 'cuda:1', 0)
    load_policy_into_vllm_instance(model, llm)

    wandb.init(
        project="a5",
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

    for idx, (input, labels, mask) in enumerate(dataloader):
        input = input.to('cuda:0')
        labels = labels.to('cuda:0')
        mask = mask.to('cuda:0')
        policy_log_probs = get_response_log_probs(model, input, labels, True)
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
            evals = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, full_dataset, sampling_params, 'temp.json')

            correct = 0
            for i in range(len(evals)):
                if evals[i]['rewards']['answer_reward'] == 1: 
                    correct += 1

            log = {'eval/accuracy': correct / len(evals),'eval_step': (idx + 1) // eval_steps}
            wandb.log(log)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/data/a5-alignment/models/Qwen2.5-Math-1.5B')
    parser.add_argument('--train_path', type=str, default='/data/a5-alignment/MATH/sft.jsonl')
    parser.add_argument('--n_examples', type=int, default=128)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_steps', type=int, default=8)

    args = parser.parse_args()

    train_sft(args.model_name, args.train_path, args.n_examples, 
              args.grad_accum_steps, args.learning_rate, args.batch_size, args.eval_steps, False)