"""
Interactive chat script for HVAC expert assistant using a fine-tuned LLM
"""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

def setup_model(model_path, tokenizer_path):
    """
    Load the model and tokenizer
    """
    print(f"Loading model from {model_path}...")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        device_map="auto",
        bnb_4bit_compute_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print("Model loaded successfully!")
    return model, tokenizer

def get_csv_context(query, reference_csv, conversation_history):
    """
    Extract relevant CSV data based on query context and conversation history
    """
    # Check entire conversation for time references
    full_conversation = query + " " + " ".join([item["content"] for item in conversation_history])
    
    # Extract timestamp from conversation if available
    time_pattern = r'(?:at|hour|time)\s*(\d+)'
    matches = re.findall(time_pattern, full_conversation)

    if matches:
        hour = int(matches[-1])  # Use the most recent mention
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

def generate_enriched_prompt(query, reference_csv, conversation_history):
    """Create an enriched prompt with CSV context and conversation history"""
    csv_context = get_csv_context(query, reference_csv, conversation_history)
    
    enriched_prompt = f"""Current building conditions (SPACE5-1):
- Indoor temperature: {csv_context['indoor_temp']:.1f}°C
- Outdoor temperature: {csv_context['outdoor_temp']:.1f}°C
- Current heating setpoint: {csv_context['heating_setpoint']:.1f}°C
- Current cooling setpoint: {csv_context['cooling_setpoint']:.1f}°C

Question: {query}

Please provide a detailed response considering these current conditions."""
    
    return enriched_prompt

def format_conversation_history(conversation_history):
    """Format the conversation history for inclusion in the prompt"""
    if not conversation_history:
        return ""
        
    formatted_history = "\n\nPrevious conversation:\n"
    for item in conversation_history:
        if item["role"] == "user":
            formatted_history += f"User: {item['content']}\n"
        else:
            formatted_history += f"Assistant: {item['content']}\n"
    
    return formatted_history

def interactive_chat(model_path, tokenizer_path, reference_csv_path):
    """
    Run an interactive chat session with the HVAC assistant
    """
    # Load model and tokenizer
    model, tokenizer = setup_model(model_path, tokenizer_path)
    
    # Load the reference CSV
    reference_csv = pd.read_csv(reference_csv_path)
    
    # Initialize conversation history
    conversation_history = []
    
    print("\n=== HVAC Expert Assistant Chat ===")
    print("Type 'exit', 'quit', or 'bye' to end the conversation")
    print("Ask questions about HVAC systems, temperature management, etc.\n")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("\nHVAC Assistant: Goodbye! Have a great day.")
            break
        
        if not user_input:
            print("Please enter a question or type 'exit' to quit.")
            continue
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_input})
        
        try:
            # Generate enriched prompt with context and history
            enriched_query = generate_enriched_prompt(user_input, reference_csv, conversation_history)
            
            # Add conversation history context if available
            history_context = format_conversation_history(conversation_history[:-1])  # Exclude current query
            
            # Create the full system prompt
            system_prompt = "You are an HVAC expert assistant. Provide helpful, detailed answers about HVAC systems, energy efficiency, and building climate control."
            
            # Construct the full prompt
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{enriched_query}{history_context} [/INST]"
            
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
            
            # Get the response and clean it
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "[/INST]" in response:
                response = response.split("[/INST]")[-1].strip()
            else:
                response = response.split("assistant\n")[-1].strip()
            
            # Print the response
            print(f"\nHVAC Assistant: {response}")
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            print(f"\nSorry, an error occurred: {str(e)}")
            print("Please try asking a different question.")

if __name__ == "__main__":
    interactive_chat(
        model_path="/home/sumanthmurthy/llama_finetuned/",
        tokenizer_path="/home/sumanthmurthy/llama_finetuned/",
        reference_csv_path="/home/sumanthmurthy/OptivAI_training/space5_training_data.csv"
    )