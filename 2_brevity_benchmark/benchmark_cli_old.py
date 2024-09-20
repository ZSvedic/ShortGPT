import typing
import click
import pandas as pd
import utils.llm_utils as llm

manual = '''CLI py app that inputs a JSONL file where each row has 2+ answers to a question and outputs a JSONL file that has the shortest answer to the question that still contains the non-trivial answer. App is called as:

python benchmark_cli.py in_jsonl out_jsonl evaluator

Where the evaluator is either "length" or the name of the HuggingFace LLM model.
For example:

python benchmark_cli.py in-short-answers.jsonl out-best.jsonl length

will compare based on length, while:

python benchmark_cli.py in-short-answers.jsonl out-best.jsonl meta-llama/Meta-Llama-3.1-8B-Instruct

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
def main_cli(in_jsonl: str, 
             out_jsonl: str, 
             evaluator: str) -> None:
    
    in_df = pd.read_json(in_jsonl, orient='records', dtype=False, lines=True)

    out_df = evaluate(in_df, evaluator)

    out_df.to_json(out_jsonl, orient='records', lines=True)

    print_summary(out_df)
   
def evaluate(in_df: pd.DataFrame, 
             evaluator: str) -> pd.DataFrame:
    ''' Gets descriptive names of in_df answer columns, finds the best answer using evaluator, 
    and returns a DataFrame with the best answer for each question. '''

    # Parse answer column names.
    prefix = "Answer-"
    answer_cols = list(in_df.columns[in_df.columns.str.startswith(prefix)])
    answer_names = [col[len(prefix):] for col in answer_cols]

    # Create output DataFrame.
    out_df = pd.DataFrame(columns=["Question", "Name-best", "Answer-best"])
    out_df["Question"] = in_df["Question"]
    # Optional ID column.
    if "ID" in in_df.columns:
        out_df["ID"] = in_df["ID"]
        # Reorder columns.
        out_df = out_df[["ID", "Question", "Name-best", "Answer-best"]]
        
    if evaluator == "length":
        evaluate_length(in_df, out_df, answer_cols, answer_names)
    else:
        tokenizer, model = llm.load_tokenizer_and_model(evaluator)
        llm_call = lambda messages: llm.batch_call_llm(tokenizer, model, messages, 50)
        evaluate_llm(llm_call, in_df, out_df, answer_cols, answer_names)

    return out_df

def evaluate_length(in_df: pd.DataFrame, 
                    out_df: pd.DataFrame, 
                    answer_cols: list, 
                    answer_names: list) -> None:
    ''' Evaluate the best answer based on the shortest length. '''
    for i, row in in_df.iterrows():
        answers = [row[col].strip() for col in answer_cols]
        id_min = min(range(len(answers)), key=lambda i: len(answers[i]))
        out_df.at[i, "Name-best"] = answer_names[id_min]
        out_df.at[i, "Answer-best"] = answers[id_min]

prompt = ''' Given a question and multiple answers, your task is to select the briefest answer that still answers the question. Examples between --- lines:
--- Example 1 ---
Input:
Q: How much is 2+3?
A1: Expression 2+3 is equal to 5.
A2: 5.
Output:
A2
--- Example 2 ---
Input:
Q: What is the color of the sky?
A1: The sky is blue.
A2: sky
Output:
A1
--- Example 3 ---
Input:
Q: What is Lorem Ipsum?
A1: Lorem Ipsum is simply dummy text used in the printing and typesetting industry.
A2: 
Output: 
A1
--- End of examples
Note that:
- The briefest answer is not the shortest if it doesn't answer the question (example 2) or is empty or invalid (example 3). 
- You always output just the ID of a question (A1, A2, etc.), nothing else, like spaces or explanations.

Given all this, what is the briefest answer to the question and answers below?
'''

def evaluate_llm(llm_call: typing.Callable[[list], list], 
                 in_df: pd.DataFrame, 
                 out_df: pd.DataFrame, 
                 answer_cols: list, 
                 answer_names: list) -> None:
    ''' Same as evaluate_length, but uses the LLM model to evaluate the best answer. '''

    messages, answers = [], []
    for _, row in in_df.iterrows():
        question = row["Question"]
        options = [row[col].strip() for col in answer_cols]
        input_text = f'Q: {question}\n' +\
            '\n'.join([f'A{i+1}: {options[i]}' for i in range(len(options))]) + \
            '\nBriefest answer: '
        messages.append([{"role": "user", "content": prompt+input_text}]) 
        answers.append(options)

    output_texts = llm_call(messages)

    for i, row in in_df.iterrows():
        id_min = int(output_texts[i].strip()[1:])-1
        out_df.at[i, "Name-best"] = answer_names[id_min]
        out_df.at[i, "Answer-best"] = answers[i][id_min]

def print_summary(out_df: pd.DataFrame) -> None:
    ''' Print a summary of the results. '''
    
    # Group by "Name-best" and perform the aggregations
    summary = out_df.groupby("Name-best").agg(
        Wins=('Name-best', 'count'),
        Win_avg_length=('Answer-best', lambda x: round(x.str.len().mean(), 1)),
        Win_example=('Answer-best', 'first'),
    )
    # Calculate winning percentages.
    summary['Wins %'] = (summary['Wins'] / len(out_df)) * 100
    # Reorder columns.
    summary = summary[['Wins', 'Wins %', 'Win_avg_length', 'Win_example']]

    print(summary)
    print(f"All winners average length: {round(out_df['Answer-best'].str.len().mean(), 1)}")
 
if __name__ == '__main__':
    main_cli()