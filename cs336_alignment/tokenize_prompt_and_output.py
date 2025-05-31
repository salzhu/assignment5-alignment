import torch 

"""
tokenize_prompt_and_output: 
- tokenizes the prompt and output strings
- constructs a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding)
"""
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    texts = [prompt + output for prompt, output in zip(prompt_strs, output_strs)]
    tokenized_texts = tokenizer(texts)

    max_length = max([len(text) for text in tokenized_texts])
    padded_texts = [text + [tokenizer.pad_token] * (max_length - len(text)) for text in tokenized_texts]

    print(torch.tensor(padded_texts['input_ids']))
    print(torch.tensor(padded_texts['input_ids']).shape)

    input_ids = tokenized_texts['input_ids'][:,:-1]
    labels = tokenized_texts['input_ids'][:,1:]
    masks = torch.tensor([[0] * len(prompt) + [1] * len(output) for prompt, output in zip(prompt_strs, output_strs)])

    return {'input_ids': input_ids, 'labels': labels, 'response_mask': masks}
