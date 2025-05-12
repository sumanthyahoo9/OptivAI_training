"""
Script to generate the responses from the fine-tuned LLM
"""
import re
import os
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def setup_dirs():
    """
    Setup the directories to save the output
    """
    os.makedirs("/home/sumanthmurthy/OptivAI_training/generated_responses_12May25/", exist_ok=True)

def generate_responses(
    model_path, 
    tokenizer_path,
    test_data_path,
    reference_csv_path,
    output_dir="generated_responses_12May25/"
):
    """Generate responses using the fine-tuned model with CSV context"""
    print(f"Loading model from {model_path}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        device_map="auto",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data
    test_data = []
    with open(test_data_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Load the reference CSV
    reference_csv = pd.read_csv(reference_csv_path)

    # Process each test example
    for i, example in enumerate(tqdm(test_data, desc="Generating responses")):
        query = example['query']
        enriched_query = generate_enriched_prompt(query, reference_csv)
        
        # Match the training format exactly
        prompt = f"<s>[INST] <<SYS>>\nYou are an HVAC expert assistant.\n<</SYS>>\n\n{enriched_query} [/INST]"
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            add_special_tokens=False
        ).to(model.device)
        
        # Try adding bad_words_ids to prevent unwanted tokens
        bad_words_ids = []
        unwanted_tokens = ['<|logistics|>', '<|welcome|>', '<|agent|>', '[WELCOME]']
        for token in unwanted_tokens:
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            if token_ids:
                bad_words_ids.append(token_ids)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask,
                bad_words_ids=bad_words_ids if bad_words_ids else None
            )
        
        # Extract the generated part
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        
        # Decode only the generated tokens
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean the response
        response = clean_response(response)
        
        # Additional check: if response still starts with unwanted content, extract actual response
        if response.startswith('<|') or 'agent_name=' in response[:50]:
            # Try to find where the actual response starts
            # Look for the first sentence that doesn't contain these tokens
            sentences = response.split('. ')
            clean_sentences = []
            for sentence in sentences:
                if not any(token in sentence for token in ['<|', '|>', 'agent_name=', '[WELCOME']):
                    clean_sentences.append(sentence)
            response = '. '.join(clean_sentences).strip()
        
        # Save the response
        output_file = os.path.join(output_dir, f"response_{i}.json")
        with open(output_file, 'w') as f:
            json.dump({
                "query": query,
                "response": response,
                "ground_truth": example.get('response', '')
            }, f, indent=2)

def clean_response(response):
    """
    Clean the response from the LLM by removing unwanted tokens and formatting
    """
    
    # First, handle the literal \n that might appear
    cleaned = response.replace('\\n', '\n')
    
    # Remove common LLM special tokens
    special_tokens = [
        '<s>', '</s>', '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>', 
        '<<RESP>>', '<|end_of_text|>', '<|endoftext|>'
    ]
    
    for token in special_tokens:
        cleaned = cleaned.replace(token, "")
    
    # Remove any logistics/agent/welcome tokens that shouldn't be there
    # Pattern 1: <|something|something>
    cleaned = re.sub(r'<\|[^>]*\|[^>]*>', '', cleaned)
    
    # Pattern 2: <|something|>
    cleaned = re.sub(r'<\|[^>]*\|>', '', cleaned)
    
    # Pattern 3: Any remaining < > brackets with pipes
    cleaned = re.sub(r'<[^>]*\|[^>]*>', '', cleaned)
    
    # Remove [WELCOME] or similar tags
    cleaned = re.sub(r'\[WELCOME\]', '', cleaned)
    cleaned = re.sub(r'\[/WELCOME\]', '', cleaned)
    
    # Remove any text that looks like role-play setup
    # This pattern catches things like "agent_name=HVAC Expert Assistant"
    cleaned = re.sub(r'agent_name=[^>]*', '', cleaned)
    
    # If the response starts with a greeting that wasn't in training, remove it
    greeting_patterns = [
        r'^Hi! I\'m the HVAC expert assistant[^.]*\.',
        r'^Hello! I\'m here to help[^.]*\.',
        r'^Welcome! [^.]*\.',
    ]
    
    for pattern in greeting_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r'^\s+', '', cleaned)
    cleaned = re.sub(r'\s+$', '', cleaned)
    
    return cleaned

def get_csv_context(query, reference_csv):
    """
    Extract relevant CSV data based on query context
    """
    # Extract timestamp from query if available
    time_pattern = r'(?:at|hour|time)\s*(\d+)'
    matches = re.findall(time_pattern, query)

    if matches:
        hour = int(matches[0])
        # Find the data points near this hour
        relevant_data = reference_csv[reference_csv['hour'] == hour]
        if len(relevant_data) > 0:
            latest_data = relevant_data.iloc[-1]
        else:
            latest_data = reference_csv.iloc[-1]
    else:
        # Return latest data as fallback
        latest_data = reference_csv.iloc[-1]
    
    # Extract values and handle NaN
    heating_sp = latest_data["action_Heating_Setpoint_RL"]
    cooling_sp = latest_data["action_Cooling_Setpoint_RL"]

    # Check for NaN and use simulated values as fallback
    nan_detected = False
    if pd.isna(heating_sp) or pd.isna(cooling_sp):
        nan_detected = True
        heating_sp = 23.25
        cooling_sp = 30.0
    
    return {
        'indoor_temp': latest_data['obs_air_temperature'],
        'outdoor_temp': latest_data['obs_outdoor_temperature'],
        'heating_setpoint': heating_sp,
        'cooling_setpoint': cooling_sp,
        "nan_detected": nan_detected
    }

def generate_enriched_prompt(query, reference_csv):
    """Create an enriched prompt with CSV context"""
    csv_context = get_csv_context(query, reference_csv)
    
    # CRITICAL: Match the training format exactly
    # During training, you just had the query with context
    # Don't add extra instructions that weren't in training
    
    context_info = f"""Current building conditions (SPACE5-1):
- Indoor temperature: {csv_context['indoor_temp']:.1f}°C
- Outdoor temperature: {csv_context['outdoor_temp']:.1f}°C
- Current heating setpoint: {csv_context['heating_setpoint']:.1f}°C
- Current cooling setpoint: {csv_context['cooling_setpoint']:.1f}°C"""
    
    # Add the NaN note if needed
    if csv_context.get('nan_detected', False):
        context_info += "\n\nNote: Using default setpoint values (23.25°C heating, 30°C cooling) as actual values were not available."
    
    # Combine context with query - matching training format
    enriched_query = f"{context_info}\n\n{query}"
    
    return enriched_query

if __name__ == "__main__":
    setup_dirs()
    generate_responses(
        model_path="/home/sumanthmurthy/llama_finetuned/",
        tokenizer_path="/home/sumanthmurthy/llama_finetuned/",
        test_data_path="/home/sumanthmurthy/OptivAI_training/test_data.jsonl",
        reference_csv_path="/home/sumanthmurthy/OptivAI_training/space5_training_data.csv"
    )