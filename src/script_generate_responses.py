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
    os.makedirs("/home/sumanthmurthy/OptivAI_training/generated_responses/", exist_ok=True)

def generate_responses(
    model_path, 
    tokenizer_path,
    test_data_path,
    reference_csv_path,
    output_dir="generated_responses"
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
        
        prompt = f"<s>[INST] <<SYS>>\nYou are an HVAC expert assistant.\n<</SYS>>\n\n{enriched_query} [/INST]"
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        response = response.split("assistant\n")[-1]
        
        # Save the response
        output_file = os.path.join(output_dir, f"response_{i}.json")
        with open(output_file, 'w') as f:
            json.dump({
                "query": query,
                "response": response,
                "ground_truth": example.get('response', '')
            }, f, indent=2)
    print(f"Generated responses saved to {output_dir}/")

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
            return {
                'indoor_temp': latest_data['obs_air_temperature'],
                'outdoor_temp': latest_data['obs_outdoor_temperature'],
                'heating_setpoint': latest_data['action_Heating_Setpoint_RL'],
                'cooling_setpoint': latest_data['action_Cooling_Setpoint_RL']
            }
    # Return latest data as fallback
    latest_data = reference_csv.iloc[-1]
    return {
        'indoor_temp': latest_data['obs_air_temperature'],
        'outdoor_temp': latest_data['obs_outdoor_temperature'],
        'heating_setpoint': latest_data['action_Heating_Setpoint_RL'],
        'cooling_setpoint': latest_data['action_Cooling_Setpoint_RL']
    }

def generate_enriched_prompt(query, reference_csv):
    """Create an enriched prompt with CSV context"""
    csv_context = get_csv_context(query, reference_csv)
    
    enriched_prompt = f"""Current building conditions (SPACE5-1):
- Indoor temperature: {csv_context['indoor_temp']:.1f}°C
- Outdoor temperature: {csv_context['outdoor_temp']:.1f}°C
- Current heating setpoint: {csv_context['heating_setpoint']:.1f}°C
- Current cooling setpoint: {csv_context['cooling_setpoint']:.1f}°C

Question: {query}

Please provide a detailed response considering these current conditions."""
    
    return enriched_prompt

if __name__ == "__main__":
    setup_dirs()
    generate_responses(
        model_path="/home/sumanthmurthy/llama_finetuned/",
        tokenizer_path="/home/sumanthmurthy/llama_finetuned/",
        test_data_path="/home/sumanthmurthy/OptivAI_training/test_data.jsonl",
        reference_csv_path="/home/sumanthmurthy/OptivAI_training/space5_training_data.csv"
    )
