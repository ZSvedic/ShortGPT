import torch
import transformers as hftf
import sys

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = 'unsloth/zephyr-sft-bnb-4bit'
    # model_name = 'meta-llama/Llama-2-7b-hf' # Will fail because of the missing chat template.

def load_tokenizer_model(model_name):
    print(f"========== Model: {model_name} ==========\n")

    # Load tokenizer
    tokenizer = hftf.AutoTokenizer.from_pretrained(
        model_name, model_max_length=128, padding_side="left", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # print("Chat template: " + tokenizer.chat_template)

    # Load model
    model = hftf.AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    print(f'Allocated GPU memory: {torch.cuda.memory_allocated() / (1024*1024):,.1f} MB')

    return tokenizer, model

def calc_model_avg_len(tokenizer, model):

    questions = [
        "How much is 2+3?",
        "What is the color of the sky?",
        "What is the capital of France?",
        "What is the boiling point of water?",
        "Who wrote 'To Kill a Mockingbird'?",
        "What is the largest planet in our solar system?",
        "What is the chemical symbol for gold?",
        "How many continents are there?",
        "What is the speed of light?",
        "Who painted the Mona Lisa?",
        "What is the smallest prime number?",
        "What is the main ingredient in guacamole?",
        "What is the square root of 64?",
        "What is the currency of Japan?",
        "Who discovered penicillin?",
        "What is the tallest mountain in the world?",
        "What is the primary language spoken in Brazil?",
        "What is the freezing point of water?",
        "What is the largest mammal?",
        "What is the capital of Japan?"
    ]

    messages = [ [{"role": "user", "content": q}] for q in questions]

    prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in messages]

    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
    inputs_tok_len = inputs["input_ids"].shape[1]
    # print(inputs_tok_len)

    # text_streamer = TextStreamer(tokenizer)
    # sequences = model.generate(input_ids = inputs[0], streamer = text_streamer, max_new_tokens = 1024, use_cache = True)
    results = model.generate(**inputs, max_new_tokens = 200, use_cache = True)
    sequences = tokenizer.batch_decode(results[:, inputs_tok_len:], skip_special_tokens=True)

    averages = []
    for answer in sequences:
        print(answer)
        print('------------------------------------------------------------------------------------')
        averages.append(len(answer))

    total_average = sum(averages)/len(averages)
    print(f"Average {model.name_or_path} answer length for {len(averages)} questions: {total_average:.2f} characters")

    return total_average

# Load the tokenizer and model:
tokenizer, model = load_tokenizer_model(model_name)
model.eval()

# Then run the verbosity test:
avg_len_before = calc_model_avg_len(tokenizer, model)
print(avg_len_before)