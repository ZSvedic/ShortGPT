import click
import time
import torch
import datasets
from datasets.utils import disable_progress_bar
import utils.llm_utils as llm

manual = '''CLI py app that inputs a JSONL file where each row has 2+ answers to a question and outputs a JSONL file that has the shortest answer to the question that still contains the non-trivial answer. App is called as:

python benchmark_cli.py in_jsonl out_jsonl evaluator continue/restart

Where the evaluator is either "length" or the name of the HuggingFace LLM model.
For example:

python benchmark_cli.py in-short-answers.jsonl out-best.jsonl length restart

will compare based on length, while:

python benchmark_cli.py in-short-answers.jsonl out-best.jsonl meta-llama/Meta-Llama-3.1-8B-Instruct continue

will use Llama-3.1-8B to evaluate the best answers.

Input JSONL fields are ID, Question, Answer-X, Answer-Y, etc. Output JSONL fields are ID, Question, Name-best, and Answer-best. Field ID contains integers starting from 101. Fields Question, Answer-X, Answer-Y, Answer are texts. The name of the answer field must begin with "Answer-"; the part after that is the descriptive name. The name-best field contains that descriptive name.

For example, for input JSONL:

{"ID": "101", "Question": "How much is 2+3?", "Answer-ChatGPT": "Expression 2+3 is equal to 5.", "Answer-Short": "5"}
{"ID": "102", "Question": "What is the color of the sky?", "Answer-ChatGPT": "The sky is blue", "Answer-Short": "sky"}

Output JSONL is:

{"ID": "101", "Question": "How much is 2+3?", "Name-best": "Short", "Answer-best": "5"}
{"ID": "102", "Question": "What is the color of the sky?", "Name-best": "ChatGPT", "Answer-best": "The sky is blue"}

As output, the app prints errors and a summary with the number of wins, average length, and a win example for each descriptive name. For the above file, the output should be:

-           Wins  Win_avg_length     Win_example

Name-best                                       

ChatGPT       1            15.0  The sky is blue

Short         1             1.0                5

All winners average length: 8.0
'''

@click.command(help=manual)
@click.argument('in_jsonl', type=click.Path(exists=True))
@click.argument('out_jsonl', type=click.Path())
@click.argument('evaluator', type=str)
@click.argument('mode', type=click.Choice(['continue', 'restart'], case_sensitive=False))
def main_click(in_jsonl: str, out_jsonl: str, evaluator: str, mode: str) -> None:
    ''' Main called by the click library (CLI). '''
    main_logic(in_jsonl, out_jsonl, evaluator, mode)

def main_logic(in_jsonl: str, out_jsonl: str, evaluator: str, mode: str) -> None:
    ''' Main called by the Click/CLI and unit tests. '''

    # Load the dataset from the JSON file.
    dataset = datasets.load_dataset('json', data_files=in_jsonl, split='train')
    # Disable progress bars for map() and similar.
    disable_progress_bar()

    # If mode is 'continue', open the file and determine the number of rows to skip.
    skip_rows = 0
    if mode == 'continue':
        try:
            with open(out_jsonl, 'r') as f:
                skip_rows = len(f.readlines())
            dataset = dataset.select(range(skip_rows, len(dataset)))
        except FileNotFoundError:
            pass
    elif mode == 'restart': 
        # If mode is 'restart', reset the output file.
        with open(out_jsonl, 'w') as f: pass

    answer_cols, answer_names = parse_answer_columns(dataset)

    # Load the tokenizer and model.
    tokenizer, model = llm.load_tokenizer_and_model(evaluator)
    print(f'Allocated GPU memory: {torch.cuda.memory_allocated() / (1024*1024):,.1f} MB')

    # Process rows in chunks and save each chunk to a JSONL file.
    process_dataset(tokenizer, model, answer_cols, answer_names, dataset, out_jsonl,
                    100, 5000, 10)

prompt_briefest_answer = ''' Given a question and multiple answers, your task is to select the briefest answer that still answers the question. Examples between --- lines:
--- Example 1 ---
Input:
Q: How much is 2+3?
A: Expression 2+3 is equal to 5.
B: 5.
Output:
B
--- Example 2 ---
Input:
Q: What is the color of the sky?
A: The sky is blue.
B: sky
Output:
A
--- Example 3 ---
Input:
Q: What is Lorem Ipsum?
A: Lorem Ipsum is simply dummy text used in the printing and typesetting industry.
B: 
Output: 
A
--- End of examples
Note that:
- The briefest answer is not the shortest if it doesn't answer the question (example 2) or is empty or invalid (example 3). 
- You always output just the ID of a question (A, B, etc.), nothing else, like spaces or explanations.

Given all this, what is the briefest answer to the question and answers below?
'''

def parse_answer_columns(in_ds: datasets.Dataset) -> tuple:
    ''' Parse answer column names. '''
    prefix = "answer-"
    answer_cols = [col for col in in_ds.column_names if col.startswith(prefix)]
    answer_names = [col[len(prefix):] for col in answer_cols]
    return answer_cols, answer_names

def get_options_and_prompts(ds: datasets.Dataset, answer_cols: list, 
                            answer_names: list) -> tuple:
    ''' Get options and prompt for all examples in the dataset. '''
    rows_options, row_prompts = [], []
    for row in ds:
        options = [row[col].strip() for col in answer_cols]
        input_text = f'Q: {row["question"]}\n' +\
            '\n'.join([f'{chr(ord("A")+i)}: {options[i]}' for i in range(len(options))]) + \
            '\nBriefest answer: '
        rows_options.append(options)
        row_prompts.append(prompt_briefest_answer + input_text)
    return rows_options, row_prompts 

def get_answer_idx(answer: str) -> int:
    ''' Get the index of the answer from the answer string. '''
    return ord(answer.strip())-ord('A')

def process_dataset(tokenizer, model, answer_cols: list, answer_names: list,
                    ds: datasets.Dataset, out_jsonl:str,
                    big_chunk_size: int, small_chunk_tokens: int, max_gen_tokens: int) -> None:
    ''' Process the input dataset in chunks and save each chunk to a JSONL file. '''

    n_rows = len(ds)
    start_time = time.time()
    current_example = 0
    print(f"Example {current_example} of {n_rows}... ", end='')
    
    for big_chunk in llm.fixed_chunker(ds, big_chunk_size):
        rows_options, row_prompts = get_options_and_prompts(big_chunk, answer_cols, answer_names)
        big_chunk = big_chunk.map(lambda _, idx: {'prompt': row_prompts[idx]}, with_indices=True)
        
        ids_min = []
        for small_chunk in llm.call_variable_chunks(
            big_chunk, tokenizer, model, big_chunk_size, small_chunk_tokens, max_gen_tokens):
            for row in small_chunk:
                ids_min.append(get_answer_idx(row['answer']))

        big_chunk = big_chunk.map(
            lambda _, idx: {'name-best': answer_names[ids_min[idx]]}, with_indices=True)
        big_chunk = big_chunk.map(
            lambda _, idx: {'answer-best': rows_options[idx][ids_min[idx]]}, with_indices=True)
        big_chunk = big_chunk.select_columns(['id', 'question', 'name-best', 'answer-best'])
        
        with open(out_jsonl, 'ab') as f:
            big_chunk.to_json(f, lines=True)

        print(f"time/sample: {(time.time()-start_time)/len(big_chunk):.2f} sec") 
        start_time = time.time()
        current_example += len(big_chunk)
        print(f"Example {current_example} of {n_rows}... ", end='')

if __name__ == '__main__':
    main_click()