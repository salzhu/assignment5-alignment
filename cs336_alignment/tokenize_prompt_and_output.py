import torch 

"""
tokenize_prompt_and_output: 
- tokenizes the prompt and output strings
- constructs a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding)
"""
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):

    tokenized_prompts = tokenizer(prompt_strs)['input_ids']
    tokenized_outputs = tokenizer(output_strs)['input_ids']

    # texts = [prompt + output for prompt, output in zip(prompt_strs, output_strs)]
    tokenized_texts = [prompt + output for prompt, output in zip(tokenized_prompts, tokenized_outputs)]

    max_length = max([len(text) for text in tokenized_texts])
    print(max_length, tokenized_texts)
    padded_texts = [text + [tokenizer.pad_token_id] * (max_length - len(text)) for text in tokenized_texts]
    print(padded_texts)

    print(torch.tensor(padded_texts))
    print(torch.tensor(padded_texts).shape)

    input_ids = torch.tensor(padded_texts)[:][:-1]
    print(input_ids.shape)
    labels = torch.tensor(padded_texts)[:][1:]
    masks = torch.tensor([[0] * len(prompt) + [1] * len(output) + [0] * (max_length - len(prompt) - len(output)) for prompt, output in zip(tokenized_prompts, tokenized_outputs)])

    return {'input_ids': input_ids, 'labels': labels, 'response_mask': masks}
