# test_simple.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/sumanthmurthy/llama_finetuned/"
tokenizer_path = "/home/sumanthmurthy/llama_finetuned/"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_4bit=True,
    device_map="auto",
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Try the exact format from training
test_prompt = "<s>[INST] <<SYS>>\nYou are an HVAC expert assistant.\n<</SYS>>\n\nWhat is HVAC? [/INST]"

inputs = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=False)
outputs = model.generate(inputs.input_ids, max_new_tokens=50, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Response: {response}")