import torch 

"""
tokenize_prompt_and_output: 
- tokenizes the prompt and output strings
- constructs a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding)
"""
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    texts = [prompt + output for prompt, output in zip(prompt_strs, output_strs)]
    tokenized_texts = tokenizer(texts)['input_ids']

    max_length = max([len(text) for text in tokenized_texts])
    print(max_length, tokenized_texts)
    padded_texts = [text + [tokenizer.pad_token_id] * (max_length - len(text)) for text in tokenized_texts]
    print(padded_texts)

    print(torch.tensor(padded_texts))
    print(torch.tensor(padded_texts).shape)

    input_ids = padded_texts[:,:-1]
    labels = padded_texts[:,1:]
    masks = torch.tensor([[0] * len(prompt) + [1] * len(output) + [0] * (max_length - len(prompt) - len(output)) for prompt, output in zip(prompt_strs, output_strs)])

    return {'input_ids': input_ids, 'labels': labels, 'response_mask': masks}
