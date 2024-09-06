import torch
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

def load_tokenizer_and_model(model_name: str):
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

def batch_call_llm(tokenizer, model, messages, max_new_tok):
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