import unittest
import pandas as pd
from benchmark_cli import main_logic

class TestEvaluateLLMFunction(unittest.TestCase):
    def test_evaluate_llm(self):
        # Run the main_cli function on a small dataset.
        main_logic('2_brevity_benchmark/in-short-answers.jsonl', 
                   '2_brevity_benchmark/out-best.jsonl', 
                   'microsoft/Phi-3-mini-4k-instruct', 
                   'restart')
        
        # Load the output file as Pandas DataFrame.
        out_df = pd.read_json("2_brevity_benchmark/out-best.jsonl", 
                             orient='records', dtype=False, lines=True) 
       
        # Check the output DataFrame.
        expected = [
            ("101", "short"), 
            ("102", "chatgpt"),
            ("103", "chatgpt"),
            ("104", "chatgpt"),
            ("105", "short"),
            ("106", "short"),
            ("107", "short"),
            ("108", "short"),
            ("109", "short"),
            ("110", "short"),
            ("111", "short"),
            ("112", "short"),
            ("113", "short"),
            ("114", "short"),
            ("115", "short"),
            ("116", "short"),
            ("117", "short"),
            ("118", "short"),
            ("119", "short"),
            ("120", "short"),
        ]
        diffs = []
        for i, row in out_df.iterrows():
            ex_id, ex_best = expected[i]
            self.assertEqual(row["question_id"], ex_id)
            if row["name-best"] != ex_best:
                diffs.append((ex_id, ex_best, row["name-best"]))
        if diffs:
            self.fail("Different: " + str(diffs))

if __name__ == '__main__':
    unittest.main()