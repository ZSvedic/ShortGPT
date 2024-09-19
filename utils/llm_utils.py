import torch
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

def load_tokenizer_and_model(model_name: str) -> tuple:
    ''' Load the HuggingFace tokenizer and model, returns both. '''
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        model_max_length=4096, 
        padding_side="left", 
        trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True,
        # Required for packing: https://huggingface.co/blog/packing-with-FA2
        attn_implementation="flash_attention_2")

    return tokenizer, model

def batch_call_llm(tokenizer, model, messages: list, max_new_tok: int) -> list:
    ''' Calls the LLM model in a batch, returns the output texts. Messages is a list of calls, 
    each item a list of messages, each message a dict with role and content. '''
    
    inputs_tokenized = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors='pt', padding=True).to('cuda')
    inputs_tok_len = inputs_tokenized.shape[1]
    # print(f"Padded input length: {inputs_tok_len}")
    inputs_att_mask = torch.cumsum(inputs_tokenized != tokenizer.eos_token_id, dim=1).bool().long()

    model.eval()
    with torch.no_grad():
        outputs_tokens = model.generate(
            inputs_tokenized, attention_mask=inputs_att_mask, 
            max_new_tokens=max_new_tok, pad_token_id=tokenizer.eos_token_id)
        
    # Output length (for each generated sequence)
    output_lengths = [len(output) for output in outputs_tokens]
    # print(f"Output lengths: {output_lengths}")

    output_texts = [tokenizer.decode(result[inputs_tok_len:], skip_special_tokens=True) 
                    for result in outputs_tokens]
                    
    return output_texts

####################################### Chunking:

from typing import Generator
import datasets # type: ignore

def fixed_chunker(dataset: datasets.Dataset, 
                  chunk_size: int) -> Generator:
    for i in range(0, len(dataset), chunk_size):
        yield dataset.select(range(i, min(i + chunk_size, len(dataset))))

def add_order_column(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(
        lambda example, idx: {'order': idx}, 
        with_indices=True)

def token_len(tokenizer, text: str) -> torch.Tensor:
    message = [{"role": "user", "content": text}] 
    tokens = tokenizer.apply_chat_template([message], add_generation_prompt=True)[0]
    return len(tokens)

def add_token_len_column(tokenizer, dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(
        lambda example: {'token_len': token_len(tokenizer, example['question'])})

def token_chunker(dataset: datasets.Dataset, 
                  chunk_tokens: int,
                  generated_tokens:int) -> Generator:
    ''' Yield chunks of examples that when prompted, tokenized, padded and generated are 
    smaller than chunk_tokens size. 
    Since examples are sorted descending by token_len, all examples in a chunk will be
    padded to the length of the first example in the chunk. '''
    start = 0
    while start < len(dataset):
        example_len = dataset[start]['token_len'] + generated_tokens
        n_chunk = min(chunk_tokens//example_len, len(dataset)-start)
        yield dataset.select(range(start, start+n_chunk))
        start += n_chunk

def chunk_call_llm(chunk: datasets.Dataset, tokenizer, model,
                   gen_tokens: int) -> None:
    messages = [[{"role": "user", "content": prompt}] 
                for prompt in chunk['prompt']]
    answers = batch_call_llm(tokenizer, model, messages, gen_tokens)
    return chunk.map(
        lambda example, idx: {'answer': answers[idx]},
        with_indices=True)

def process_variable_chunks(dataset: datasets.Dataset, 
                            tokenizer,
                            model,
                            big_chunk_size: int,
                            small_chunk_tokens: int,
                            max_gen_tokens: int) -> Generator:

    for chunk in fixed_chunker(dataset, big_chunk_size):
        chunk = add_token_len_column(tokenizer, chunk)
        chunk = add_order_column(chunk)
        chunk = chunk.sort('token_len', reverse=True)
        chunk = datasets.concatenate_datasets(
            [chunk_call_llm(small_chunk, tokenizer, model, max_gen_tokens) 
             for small_chunk in token_chunker(chunk, small_chunk_tokens, max_gen_tokens)] )
        chunk = chunk.sort('order')
        yield chunk
