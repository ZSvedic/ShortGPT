from datasets import load_dataset
from trl import DPOConfig, DPOTrainer, LogCompletionsCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Trainer, TrainerCallback
from typing import Optional
import torch
import wandb
from trl.models.utils import unwrap_model_for_generation


class LogCompletionsLengthCallback(TrainerCallback):
    def __init__(self, trainer: Trainer, num_prompts: Optional[int] = None, freq: Optional[int] = None):
        self.trainer = trainer
        self.freq = freq
        self._last_logged_step = -1
        self.eval_dataset = self.trainer.eval_dataset
        self.eval_dataset = self.eval_dataset.select(range(num_prompts))

    def on_step_end(self, args, state, control, **kwargs):
        # Only log once per step (this method may be called multiple times)
        if state.global_step == self._last_logged_step:
            return

        # Only log every `freq` steps (if no `freq` is provided, log every `eval_steps` steps)
        freq = self.freq or state.eval_steps
        if state.global_step % freq != 0:
            return

        tokenizer = kwargs["processing_class"]
        tokenizer.padding_side = "left"
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped
        completion_lens = []
        with accelerator.split_between_processes(self.eval_dataset["prompt_input_ids"]) as prompts:
            with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
                for prompt_ids in prompts:
                    prompt_ids = torch.tensor([prompt_ids], device=unwrapped_model.device)
                    generations = unwrapped_model.generate(
                        prompt_ids, generation_config=GenerationConfig(max_new_tokens=100)
                    )
                    completion_lens.append(len(generations[0]) - len(prompt_ids[0]))

        # Build the data to log
        if self.trainer.accelerator.is_main_process:
            wandb.log({"completions_len": sum(completion_lens) / len(completion_lens)}, step=state.global_step)

        # Save the last logged step, so we don't log the same completions multiple times
        self._last_logged_step = state.global_step


def main():
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    dataset = load_dataset("ZSvedic/gpt4o-arena-brevity-dpo")
    dataset["test"] = dataset["test"].select(range(20))

    def make_conv(example):
        return {
            "prompt": [{"role": "user", "content": example["prompt"]}],
            "chosen": [{"role": "assistant", "content": example["chosen"]}],
            "rejected": [{"role": "assistant", "content": example["rejected"]}],
        }

    dataset = dataset.map(make_conv)

    training_args = DPOConfig(
        output_dir="Qwen2-0.5B-DPO",
        logging_steps=5,
        eval_steps=5,
        eval_strategy="steps",
        gradient_accumulation_steps=8,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    callback = LogCompletionsLengthCallback(trainer, num_prompts=16)
    trainer.add_callback(callback)
    trainer.train()


if __name__ == "__main__":
    main()