import click
import random
import torch
import datasets # type: ignore
import utils.llm_utils as llm
import utils.utils as utils

manual = '''CLI py app that gets input dataset name, question column name, and outputs a JSONL file with the dataset containing normal and short answers. App is called as:

python dataset_cli.py input_dataset question_column output_jsonl continue/restart

For example:

python dataset_cli.py lmsys/chatbot_arena_conversations conversation_a chatbot_arena_long_short_dataset.jsonl continue

App uses Pandas to load and save the dataset. 
'''

@click.command(help=manual)
@click.argument('input_dataset', type=str)
@click.argument('question_column', type=str)
@click.argument('output_jsonl', type=str)
@click.argument('mode', type=click.Choice(['continue', 'restart'], case_sensitive=False))
def main_cli(input_dataset: str, question_column: str, output_jsonl: str, mode: str):
    # Load the dataset from HuggingFace.
    dataset = datasets.load_dataset(input_dataset)['train']
    questions = [row[question_column][0]['content'] for row in dataset]
    # questions = [dataset[i][question_column][0]['content'] for i in range(360)] # Only N rows.

    # Load the tokenizer and model.
    # model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    # model_name = 'microsoft/Phi-3-small-128k-instruct'
    model_name = 'microsoft/Phi-3-mini-4k-instruct'
    tokenizer, model = llm.load_tokenizer_and_model(model_name)
    print(f'Allocated GPU memory: {torch.cuda.memory_allocated() / (1024*1024):,.1f} MB')
    llm_call = lambda messages, new_tokens: \
        llm.batch_call_llm(tokenizer, model, messages, new_tokens)
    
    # If mode is 'continue', open the file and determine the number of rows to skip.
    skip_rows = 0
    if mode == 'continue':
        try:
            with open(output_jsonl, 'r') as f:
                skip_rows = len(f.readlines())
        except FileNotFoundError:
            pass
    elif mode == 'restart': 
        # If mode is 'restart', reset the output file.
        with open(output_jsonl, 'w') as f: pass

    # Process questions in chunks and save each chunk to a JSONL file.
    smart_chunker(llm_call, questions, skip_rows, output_jsonl)

# Word limits for normal and brief answers.
normal_word_limit = 120
brief_word_limit = 20
# A word is "between 5 and 6.5 characters per word including spaces and punctuation":
# https://charactercounter.com/characters-to-words
normal_max_ch_soft = normal_word_limit * 6
brief_max_ch_soft = brief_word_limit * 6
# Hard limit adds 20% buffer and divides by 4 to get LLM token limit.
# https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
normal_max_tokens_hard = normal_max_ch_soft * 1.2 / 4
brief_max_tokens_hard = brief_max_ch_soft * 1.2 / 4

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

def chunk_generator(llm_call, questions, chunk_size=25):
    ''' Generator that yields chunks of questions, normal answers, and brief answers. '''
    for i in range(0, len(questions), chunk_size):
        chunk_questions = questions[i:i+chunk_size]
        normal_answers = llm_normal_answers(llm_call, chunk_questions)
        brief_answers = llm_brief_answers(llm_call, chunk_questions, normal_answers)
        for q, na, ba in zip(chunk_questions, normal_answers, brief_answers):
            yield {"Question": q, "Answer-normal": na, "Answer-short": ba}

def smart_chunker(llm_call, questions:list, skip_rows:int, out_jsonl:str, 
                  q_chunk_big=100, q_chunk_small=20, n_chunk_max_ch=22_000):
    ''' Similar to chunk_generator, but minimizes the inefficiency of long questions and 
    long outputs (from the first LLM call) adding padding to shorter questions and outputs. 
    It works by breaking into smaller chunks and chunks that don't exceed chunk_max_ch. '''

    # Positions of columns in table.
    p_id, p_question, p_normal, p_size, p_brief = 0, 1, 2, 3, 4 

    # Slice into big chunks of questions.
    for i in range(skip_rows, len(questions), q_chunk_big):

        # Create table with id, question, normal, size, and brief columns.
        table = [[id, question, None, None, None] 
                for id, question in enumerate(questions[i:i+q_chunk_big])]
        chunk_len = len(table) # Last chunk can be smaller than q_chunk_big.
        
        timer = utils.Timer(f"Question {i} of {len(questions)}...")
        with timer:

            # Slice into small chunks of questions.
            for j in range(0, chunk_len, q_chunk_small):
                # Call LLM for normal answers of the small chunks.
                table_slice = table[j:j+q_chunk_small]
                normal_answers = llm_normal_answers(llm_call, [row[p_question] for row in table_slice])
                for k, normal in enumerate(normal_answers):
                    table_slice[k][p_normal] = normal
            
            # Estimate the character size of brief queries.
            brief_query_const = len(brief_prompt) + brief_max_ch_soft
            for row in table:
                row[p_size] = brief_query_const + len(row[p_question]) + len(row[p_normal]) 

            # Sort by size.
            table.sort(key=lambda x: x[p_size])

            # Create chunks of brief calls that don't exceed n_chunk_max_ch.
            n_ch = 0
            start = 0
            for j in range(chunk_len):
                n_ch += table[j][p_size]
                if n_ch>n_chunk_max_ch or j==chunk_len-1:
                    # Call LLM for brief answers of the chunk.
                    table_slice = table[start:j+1]
                    brief_answers = llm_brief_answers(
                        llm_call, 
                        [row[p_question] for row in table_slice], 
                        [row[p_normal] for row in table_slice])
                    for k, brief in enumerate(brief_answers):
                        table_slice[k][p_brief] = brief
                    # Reset counters.
                    n_ch = 0
                    start = j+1

            # Sort by id.
            table.sort(key=lambda x: x[p_id])

            # Create a Dataset from table.
            out_ds = datasets.Dataset.from_list(
                [{"Question": row[p_question], 
                  "Answer-short": row[p_brief], 
                  "Answer-normal": row[p_normal]}
                 for row in table])
            # Append to JSONL.
            with open(out_jsonl, 'ab') as f:
                out_ds.to_json(f, lines=True)   

        print(f"Time/sample: {timer.last_elapsed_time/chunk_len:.2f} sec")            

def llm_normal_answers(llm_call, questions):
    messages = [ [{"role": "user", 
                   "content": normal_prompt+q}] 
                for q in questions ]
    return llm_call(messages, normal_max_tokens_hard)

def llm_brief_answers(llm_call, questions, normal_answers):
    messages = [ [{"role": "user", 
                   "content": f"{brief_prompt}Question: {q}\nNormal answer: {na}"}] 
                for q, na in zip(questions, normal_answers) ]
    return llm_call(messages, brief_max_tokens_hard)
        
if __name__ == '__main__':
    main_cli()