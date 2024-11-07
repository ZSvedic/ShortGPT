import click
import torch
import time
import datasets
from datasets.utils import disable_progress_bar
import utils.llm_utils as llm
import utils.utils as utils

manual = '''CLI py app that gets input dataset name and outputs a JSONL file with the dataset containing normal and short answers. App is called as:

python dataset_cli.py input_dataset output_jsonl continue/restart

For example:

python dataset_cli.py lmsys/chatbot_arena_conversations chatbot_arena_long_short_dataset.jsonl continue
'''

@click.command(help=manual)
@click.argument('input_dataset', type=str)
@click.argument('output_jsonl', type=str)
@click.argument('mode', type=click.Choice(['continue', 'restart'], case_sensitive=False))
def main_cli(input_dataset: str, output_jsonl: str, mode: str):
    # Load the dataset from HuggingFace.
    dataset = datasets.load_dataset(input_dataset)['train']
    # Disable progress bars for map() and similar.
    disable_progress_bar()

    # If mode is 'continue', open the file and determine the number of rows to skip.
    skip_rows = 0
    if mode == 'continue':
        try:
            with open(output_jsonl, 'r') as f:
                skip_rows = len(f.readlines())
            dataset = dataset.select(range(skip_rows, len(dataset)))
        except FileNotFoundError:
            pass
    elif mode == 'restart': 
        # If mode is 'restart', reset the output file.
        with open(output_jsonl, 'w') as f: pass

    # Load the tokenizer and model.
    # model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    # model_name = 'microsoft/Phi-3-small-128k-instruct'
    model_name = 'microsoft/Phi-3-mini-4k-instruct'
    tokenizer, model = llm.load_tokenizer_and_model(model_name)
    print(f'Allocated GPU memory: {torch.cuda.memory_allocated() / (1024*1024):,.1f} MB')

    # Process rows in chunks and save each chunk to a JSONL file.
    process_dataset(tokenizer, model, dataset, output_jsonl)

# Word limits for normal and brief answers.
normal_word_limit = 120
brief_word_limit = 20
# A word is "between 5 and 6.5 characters per word including spaces and punctuation":
# https://charactercounter.com/characters-to-words
normal_max_ch_soft = normal_word_limit * 6
brief_max_ch_soft = brief_word_limit * 6
# Hard limit adds 20% buffer and divides by 4 to get LLM token limit.
# https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
normal_max_tokens_hard = int(normal_max_ch_soft * 1.2 / 4)
brief_max_tokens_hard = int(brief_max_ch_soft * 1.2 / 4)

normal_prompt = f'''Answer the user prompt below "---" line. Never exceed {normal_max_ch_soft} characters / {normal_word_limit} words.
---
'''

brief_prompt = f'''Given the user prompt and a normal answer, generate a brief answer. A brief answer should be as short as possible but still answer the question and give relevant information. Never exceed {brief_max_ch_soft} characters / {brief_word_limit} words.
Examples between --- lines:
--- Example 1 ---
Your input:
Question: How much is 2+3?
Normal answer: Expression 2+3 is equal to 5.
Your output:
5
--- Example 2 ---
Your input:
Question: What is the color of the sky?
Normal answer: The sky is blue.
Your output:
Blue
--- End of examples

Considering all the above, give a brief answer to the prompt and normal answer below:
'''

def get_brief_prompt(example) -> str:
    q = example['question']
    na = example['answer-normal']
    return f"{brief_prompt}Question: {q}\nNormal answer: {na}"

def process_dataset(tokenizer, model, 
                    ds: datasets.Dataset, out_jsonl:str) -> None:
    ''' Process the 'lmsys/chatbot_arena_conversations' dataset in chunks and save 
    each chunk to a JSONL file. '''

    ds = ds.select_columns(['question_id', 'conversation_a']) 
    ds = ds.rename_column('question_id', 'question-id')
    ds = ds.map(lambda example: {'question': example['conversation_a'][0]['content']}) 
    ds = ds.map(lambda example: {'prompt': normal_prompt + example['question']})

    n_rows = len(ds)
    start_time = time.time()
    current_example = 0
    print(f"Example {current_example} of {n_rows}... ", end='')

    for norm_c in llm.call_variable_chunks(
        ds, tokenizer, model, 100, 5000, normal_max_tokens_hard):

        norm_c = norm_c.select_columns(['question-id', 'question', 'answer'])
        norm_c = norm_c.rename_column('answer', 'answer-normal')
        norm_c = norm_c.map(lambda example: {'prompt': get_brief_prompt(example)})
        
        for brief_c in llm.call_variable_chunks(
            norm_c, tokenizer, model, 100, 5000, brief_max_tokens_hard):
            brief_c = brief_c.select_columns(['question-id', 'question', 'answer', 'answer-normal'])
            brief_c = brief_c.rename_column('answer', 'answer-short')
            with open(out_jsonl, 'ab') as f:
                brief_c.to_json(f, lines=True)

        print(f"time/sample: {(time.time()-start_time)/len(norm_c):.2f} sec") 
        start_time = time.time()
        current_example += len(norm_c)
        print(f"Example {current_example} of {n_rows}... ", end='')
        
if __name__ == '__main__':
    main_cli()