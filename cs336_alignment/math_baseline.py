import json 
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn

sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)
# drgrpo_reward_fn = r1_zero_reward_fn

"""
{"problem": "Suppose that the least common multiple of the first $25$ positive integers is equal to 
$26A7114B4C0$. Find $100 \\times A + 10 \\times B + C$.", "level": "Level 5", "subject": "Number Theory", 
"unique_id": "test/number_theory/475.json", "answer": "740"}
"""

def evaluate_vllm(
        vllm_model, 
        reward_fn, 
        prompts, 
        answers,
        full_problems,
        eval_sampling_params, 
        savepath="rewards.json"
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    evals = []
    for i in range(len(outputs)):
        rewards = reward_fn(outputs[i], answers[i])
        eval = {
            "prompt": prompts[i], 
            "response": outputs[i].outputs[0].text, 
            "rewards": rewards, 
            "full": full_problems[i]
        }
        evals.append(eval)
        
    with open(savepath, 'w') as f:
        json.dump(evals, f) # indent for readability

    return evals 

def evaluate_math(model_name, dataset, prompt, savepath, is_prompt=False, replacement="{{question}}"): 

    if is_prompt == False:
        with open(prompt, "r") as file:
            prompt = file.read()

    prompts = []
    answers = []

    with open(dataset, 'r') as file:
        for line in file:
            data_dict = json.loads(line)
            print(data_dict)
        data = json.load(file)
        problem = data["problem"]
        prompts.append(problem.replace(replacement, problem))
        answers.append(data["answer"])
        
    print(prompts[:3])

    llm = LLM(model=model_name)

    return evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, data, sampling_params, savepath)

if __name__ == '__main__':
    model_path = '/data/a5-alignment/models/Qwen2.5-Math-1.5B'
    dataset_path = '/data/a5-alignment/MATH/validation.jsonl'
    prompt_path = '/home/c-salzhu/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt'
    evals = evaluate_math(model_path, dataset_path, prompt_path, '/data/c-salzhu/qwen_2.5_math_1.5b_MATH_baseline.json')

    format_answer = 0 
    format_noanswer = 0 
    noformat_noanswer = 0 

    for eval in evals: 
        if eval['rewards']['format_reward'] == 1 and eval['rewards']['answer_reward'] == 1:
            format_answer += 1
        elif eval['rewards']['format_reward'] == 1 and eval['rewards']['answer_reward'] == 0:
            format_noanswer += 1
        elif eval['rewards']['format_reward'] == 0 and eval['rewards']['answer_reward'] == 0:
            noformat_noanswer += 1
    print(f'format1, answer1: {format_answer} | format1, answer0: {format_noanswer} | format0, answer0: {noformat_noanswer}')

    print('******************************************************************')
    count = 0 
    for eval in evals: 
        if eval['rewards']['format_reward'] == 0: 
            print(eval)
            print('--------------------------------------------------------------------')
            count += 1 
        if count > 10:
            break 

    print('******************************************************************')
    count = 0 
    for eval in evals: 
        if eval['rewards']['format_reward'] == 1 and eval['rewards']['answer_reward'] == 0: 
            print(eval)
            print('--------------------------------------------------------------------')
            count += 1 
        if count > 10:
            break 