import click
import csv
import pandas as pd # type: ignore
import utils.llm_utils as llm

manual = '''CLI py app that inputs a CSV file where each row has 2+ answers to a question and outputs a CSV file that has the shortest answer to the question that still contains the non-trivial answer. App is called as:

python benchmark_cli.py in_csv out_csv evaluator

Where the evaluator is either "length" or the name of the HuggingFace LLM model.
For example:

python benchmark_cli.py chatgpt-short-answers.csv chatgpt-short-best.csv length

will compare based on length, while:

python benchmark_cli.py chatgpt-short-answers.csv chatgpt-short-best.csv meta-llama/Meta-Llama-3.1-8B-Instruct

will use Llama-3.1-8B to evaluate the best answers.

Input CSV columns are ID, Question, Answer-X, Answer-Y, etc. Output CSV columns are ID, Question, Name-best, and Answer-best. Column ID contains integers starting from 101. Columns Question, Answer-X, Answer-Y, Answer are texts. The name of the answer column must begin with "Answer-"; the part after that is the descriptive name. The name-best column contains that descriptive name. 

For example, for input CSV: 

"ID", "Question", "Answer-ChatGPT", "Answer-Short" 

"101", "How much is 2+3?", "Expression 2+3 is equal to 5.", "5" 

"102", "What is the color of the sky?", "The sky is blue", "sky"  

Output CSV is:

"ID", "Question", "Name-best", "Answer-best"

"101", "How much is 2+3?", "Short", "5"

"102", "What is the color of the sky?", "ChatGPT", "The sky is blue"

As output, the app prints errors and a summary with the number of wins, average length, and a win example for each descriptive name. For the above file, the output should be:

-           Wins  Win_avg_length     Win_example

Name-best                                       

ChatGPT       1            15.0  The sky is blue

Short         1             1.0                5

All winners average length: 8.0
'''

@click.command(help=manual)
@click.argument('in_csv', type=click.Path(exists=True))
@click.argument('out_csv', type=click.Path())
@click.argument('evaluator', type=str)
def main_cli(in_csv: str, out_csv: str, evaluator: str):
    in_df = read_csv(in_csv) 

    out_df = evaluate(in_df, evaluator)

    out_df.to_csv(out_csv, index=False, quoting=csv.QUOTE_ALL, quotechar='"')

    # Group by "Name-best" and perform the aggregations
    summary = out_df.groupby("Name-best").agg(
        Wins=('Name-best', 'count'),
        Win_avg_length=('Answer-best', lambda x: round(x.str.len().mean(), 1)),
        Win_example=('Answer-best', 'first'),
    )

    print(summary)
    print(f"All winners average length: {round(out_df['Answer-best'].str.len().mean(), 1)}")

def read_csv(in_csv: str) -> pd.DataFrame:
    in_df = pd.read_csv(
        in_csv, skipinitialspace=True, quoting=csv.QUOTE_ALL, quotechar='"', 
        keep_default_na=False, dtype=str)
    in_df.rename(columns=lambda x: x.strip(), inplace=True) # Pandas spaces outside quotes fix.
    return in_df
    
def evaluate(in_df, evaluator: str):
    ''' Gets descriptive names of in_df answer columns, finds the best answer using evaluator, 
    and returns a DataFrame with the best answer for each question. '''

    # Parse answer column names.
    prefix = "Answer-"
    answer_cols = list(in_df.columns[in_df.columns.str.startswith(prefix)])
    answer_names = [col[len(prefix):] for col in answer_cols]

    # Create output DataFrame.
    out_df = pd.DataFrame(columns=["ID", "Question", "Name-best", "Answer-best"])
    out_df["ID"] = in_df["ID"]
    out_df["Question"] = in_df["Question"]

    if evaluator == "length":
        evaluate_length(in_df, out_df, answer_cols, answer_names)
    else:
        tokenizer, model = llm.load_tokenizer_and_model(evaluator)
        llm_call = lambda messages: llm.batch_call_llm(tokenizer, model, messages, 50)
        evaluate_llm(llm_call, in_df, out_df, answer_cols, answer_names)

    return out_df

def evaluate_length(in_df, out_df, answer_cols, answer_names):
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
A2: Sky.
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

def evaluate_llm(llm_call, in_df, out_df, answer_cols, answer_names):
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


if __name__ == '__main__':
    main_cli()
