import unittest
import pandas as pd
from benchmark_cli import evaluate

class TestEvaluateLLMFunction(unittest.TestCase):
    def test_evaluate_llm(self):
        in_df = pd.read_json("2_brevity_benchmark/in-short-answers.jsonl", 
                             orient='records', dtype=False, lines=True) 
        out_df = evaluate(in_df, "meta-llama/Meta-Llama-3.1-8B-Instruct")
        expected = [
            ("101", "Short"), 
            ("102", "ChatGPT"),
            ("103", "ChatGPT"),
            ("104", "ChatGPT"),
            ("105", "ChatGPT"),
            ("106", "Short"),
            ("107", "Short"),
            ("108", "Short"),
            ("109", "Short"),
            ("110", "Short"),
            ("111", "Short"),
            ("112", "Short"),
            ("113", "Short"),
            ("114", "Short"),
            ("115", "Short"),
            ("116", "Short"),
            ("117", "Short"),
            ("118", "Short"),
            ("119", "Short"),
            ("120", "Short"),
        ]
        for i, row in out_df.iterrows():
            ex_id, ex_best = expected[i]
            self.assertEqual(row["ID"], ex_id)
            self.assertEqual(row["Name-best"], ex_best)

if __name__ == '__main__':
    unittest.main()