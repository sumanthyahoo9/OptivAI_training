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
    
    # CRITICAL: Make sure the tokenizer has proper tokens set
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
        
        # CRITICAL CHANGE: Match EXACTLY the format used during training
        # Your training script used: "<s>[INST] <<SYS>>\nYou are an HVAC expert assistant.\n<</SYS>>\n\n{example['query']} [/INST] {example['response']}</s>"
        # So we need to match this format exactly
        prompt = f"<s>[INST] <<SYS>>\nYou are an HVAC expert assistant.\n<</SYS>>\n\n{enriched_query} [/INST]"
        
        # Tokenize without adding special tokens (we've already added them)
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            add_special_tokens=False  # Changed to False since we manually added <s>
        ).to(model.device)
        
        # Debug print to see what we're sending
        if i == 0:  # Only for first example
            print(f"First prompt being sent: {prompt[:200]}...")
            print(f"Input shape: {inputs.input_ids.shape}")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,  # Use input_ids directly
                max_new_tokens=1024,
                temperature=0.7,  # Increased for more variability
                do_sample=True,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask
            )
        
        # Extract only the generated tokens (not the input)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        
        # Decode only the generated part
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # If still empty, try alternative decoding
        if not response.strip():
            # Try decoding the full output and extracting after [/INST]
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
            if "[/INST]" in full_output:
                response = full_output.split("[/INST]")[-1].strip()
                # Remove any trailing </s> or other tokens
                response = response.replace("</s>", "").strip()
        
        # Final cleaning
        response = clean_response(response)
        
        # Debug information for empty responses
        if not response.strip():
            print(f"\nWarning: Empty response for query {i}")
            print(f"Query: {query[:100]}...")
            print(f"Generated tokens: {generated_tokens.tolist()[:10]}...")
            print(f"Full decoded output: {tokenizer.decode(outputs[0])[:200]}...")
        
        # Save the response
        output_file = os.path.join(output_dir, f"response_{i}.json")
        with open(output_file, 'w') as f:
            json.dump({
                "query": query,
                "response": response,
                "ground_truth": example.get('response', '')
            }, f, indent=2)
    
    print(f"Generated responses saved to {output_dir}/")

def clean_response(response):
    """
    Clean the response from the LLM by removing certain tokens
    """
    # Remove special tokens
    special_tokens = ['<s>', '</s>', '[INST]', '[/INST]', '<<SYS>>', '<</SYS>>', '<<RESP>>', '<|end_of_text|>']
    
    cleaned = response
    for token in special_tokens:
        cleaned = cleaned.replace(token, "")
    
    # Remove any repeated newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Strip whitespace
    return cleaned.strip()

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