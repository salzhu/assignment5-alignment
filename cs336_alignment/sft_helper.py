import torch 

"""
4.2 tokenize_prompt_and_output: 
- tokenizes the prompt and output strings
- constructs a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding)
"""
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):

    tokenized_prompts = tokenizer(prompt_strs)['input_ids']
    tokenized_outputs = tokenizer(output_strs)['input_ids']

    tokenized_texts = [prompt + output for prompt, output in zip(tokenized_prompts, tokenized_outputs)]

    max_length = max([len(text) for text in tokenized_texts])
    padded_texts = [text + [tokenizer.pad_token_id] * (max_length - len(text)) for text in tokenized_texts]

    input_ids = torch.tensor(padded_texts)[:,:-1]
    labels = torch.tensor(padded_texts)[:,1:]
    masks = torch.tensor([[False] * (len(prompt) - 1) + [True] * len(output) + [False] * (max_length - len(prompt) - len(output)) for prompt, output in zip(tokenized_prompts, tokenized_outputs)])

    return {'input_ids': input_ids, 'labels': labels, 'response_mask': masks}

"""
4.2 compute_entropy
- computes the per-token entropy of next-token predictions
"""
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logitnorm = torch.logsumexp(logits, dim=-1) # over the vocab 
    # print(logits.shape, logitnorm.shape)
    logprobs = logits - torch.unsqueeze(logitnorm,dim=-1) # normalize the log 
    return torch.sum(-1 * logprobs * torch.exp(logprobs), dim=-1)

"""
4.2 get_response_log_probs: 
- gets per-token conditional log-probabilities (given the previous tokens) from a causal language model
- optionally the entropy of the model’s next-token distribution
"""
def get_response_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    logits = model(input_ids).logits # batch size x seq len x vocab size
    logprobs = torch.log(torch.nn.functional.softmax(logits, dim=-1)) #, dim=
    logprobs = torch.gather(logprobs, -1, torch.unsqueeze(labels, dim=-1))
    logprobs = torch.squeeze(logprobs)

    if not return_token_entropy:
        return {'log_probs': logprobs}
    return {'log_probs': logprobs, 'token_entropy': compute_entropy(logits)}

"""
4.2 masked_normalize
- sums over tensor elements and
- normalizes by a constant while respecting a boolean mask
"""
def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    
    tensor = tensor * mask 
    sum_tensor = torch.sum(tensor, dim=dim)
    return sum_tensor / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    # loss = torch.nn.functional.cross_entropy(policy_log_probs)
    
    logprobs = masked_normalize(policy_log_probs, response_mask, normalize_constant)
    # loss = compute_entropy(logprobs) / gradient_accumulation_steps
    loss = torch.mean(logprobs)  / gradient_accumulation_steps
    loss.backward() 

    return loss, loss