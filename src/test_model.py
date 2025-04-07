"""
Test the fine-tuned model
"""
import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model(model_path):
    """
    Load the fine-tuned HVAC control model
    """
    print(f"Loading fine-tuned model from {model_path}...")
    
    # Load the PEFT configuration
    config = PeftConfig.from_pretrained(model_path)
    
    # Load the base model
    print(f"Loading base model: {config.base_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.float16
    )
    
    # Load the PEFT adapter
    print("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(model, model_path)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    return model, tokenizer

def load_blueprint_context(context_path):
    """
    Load the building blueprint context
    """
    try:
        with open(context_path, 'r') as f:
            context = f.read()
        print(f"Loaded blueprint context: {len(context.split())} words")
        return context
    except Exception as e:
        print(f"Error loading blueprint context: {e}")
        return ""

def generate_hvac_response(model, tokenizer, query, blueprint_context="", max_new_tokens=512):
    """
    Generate a response for an HVAC control query using the fine-tuned model
    """
    # Format the query with the blueprint context
    if blueprint_context:
        formatted_query = f"Building Blueprint Context:\n{blueprint_context}\n\nUser Query: {query}"
    else:
        formatted_query = query
    
    # Format the prompt according to Llama model instruction format
    prompt = f"<s>[INST] {formatted_query} [/INST]"
    
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # Decode and return only the generated part (not the prompt)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

def process_query_file(model, tokenizer, query_file, blueprint_context="", output_file=None):
    """
    Process a file containing multiple queries, one per line
    """
    try:
        with open(query_file, 'r') as f:
            queries = f.readlines()
        
        responses = []
        for i, query in enumerate(queries):
            query = query.strip()
            if not query:  # Skip empty lines
                continue
                
            print(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            response = generate_hvac_response(model, tokenizer, query, blueprint_context)
            responses.append({"query": query, "response": response})
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(responses, f, indent=2)
            print(f"Saved responses to {output_file}")
        
        return responses
    
    except Exception as e:
        print(f"Error processing query file: {e}")
        return []

def interactive_mode(model, tokenizer, blueprint_context=""):
    """
    Run the model in interactive mode, taking queries from the console
    """
    print("\nHVAC Control Assistant - Interactive Mode")
    print("Enter 'quit', 'exit', or 'q' to exit\n")
    
    while True:
        query = input("\nEnter your HVAC control query: ")
        
        if query.lower() in ["quit", "exit", "q"]:
            print("Exiting interactive mode")
            break
        
        if not query:
            continue
        
        print("\nGenerating response...")
        response = generate_hvac_response(model, tokenizer, query, blueprint_context)
        print("\nResponse:")
        print("---------")
        print(response)
        print("---------")

def main():
    parser = argparse.ArgumentParser(description="HVAC Fine-Tuned Model Inference")
    parser.add_argument("--model_path", type=str, default="hvac_finetuned_model/final", 
                        help="Path to the fine-tuned model")
    parser.add_argument("--blueprint_context", type=str, default="blueprint_context.txt",
                        help="Path to the building blueprint context file")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--query_file", type=str, help="File containing multiple queries, one per line")
    parser.add_argument("--output_file", type=str, help="Output file for query results")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Load the model
    model, tokenizer = load_model(args.model_path)
    
    # Load the blueprint context
    blueprint_context = ""
    if os.path.exists(args.blueprint_context):
        blueprint_context = load_blueprint_context(args.blueprint_context)
    
    # Process a single query
    if args.query:
        response = generate_hvac_response(model, tokenizer, args.query, blueprint_context)
        print("\nResponse:")
        print("---------")
        print(response)
        print("---------")
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump({"query": args.query, "response": response}, f, indent=2)
            print(f"Saved response to {args.output_file}")
    
    # Process a file containing multiple queries
    elif args.query_file:
        process_query_file(model, tokenizer, args.query_file, blueprint_context, args.output_file)
    
    # Run in interactive mode
    elif args.interactive:
        interactive_mode(model, tokenizer, blueprint_context)
    
    # If no mode specified, run in interactive mode
    else:
        interactive_mode(model, tokenizer, blueprint_context)

if __name__ == "__main__":
    main()