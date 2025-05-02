"""
The main file to commence a training job
This script:
Reads data from the relevant sources
Loads the relevant models
Trains/Fine-tunes the model.
Saves checkpoints.
"""
import os
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training, 
    get_peft_model
)
from tqdm import tqdm

# Configuration parameters, change accordingly
MODEL_NAME = "meta-llama/Llama-3.1-8B"
OUTPUT_DIR = "hvac_finetuned_model"
DATASET_PATH = "fine_tuning_data.csv"
BLUEPRINT_CONTEXT_PATH = "blueprint_context.txt"  # OCR extracted blueprint data
MAX_LENGTH = 2048  # Maximum sequence length
LORA_R = 16  # LoRA attention dimension
LORA_ALPHA = 32  # LoRA alpha parameter
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load blueprint context
def load_blueprint_context(file_path):
    """Load building blueprint context from file"""
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Warning: Blueprint context file {file_path} not found. Proceeding without blueprint context.")
        return ""

# Load and preprocess the fine-tuning dataset
def preprocess_dataset(csv_path, blueprint_context=""):
    """
    Load and preprocess the HVAC fine-tuning dataset
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Check if the necessary columns exist
    required_columns = ['query', 'response']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"The dataset is missing the required column: {col}")
    
    print(f"Dataset loaded with {len(df)} examples")
    
    # Add blueprint context to each query if available
    if blueprint_context:
        processed_data = []
        for idx, row in df.iterrows():
            # Format with blueprint context
            processed_data.append({
                "query": f"Building Blueprint Context:\n{blueprint_context}\n\nUser Query: {row['query']}",
                "response": row['response']
            })
    else:
        processed_data = [{"query": row['query'], "response": row['response']} for idx, row in df.iterrows()]
    
    # Convert to Hugging Face dataset
    dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
    
    return dataset

# Format examples for instruction fine-tuning
def format_instruction(example):
    """Format each example as an instruction for the LLM"""
    return f"<s>[INST] {example['query']} [/INST] {example['response']}</s>"

# Tokenize the dataset
def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset for training"""
    formatted_texts = [format_instruction(example) for example in dataset]
    
    # Tokenize with padding to max_length
    tokenized_data = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Prepare the labels (same as input_ids, as we're doing causal LM training)
    tokenized_data["labels"] = tokenized_data["input_ids"].clone()
    
    return tokenized_data

def main():
    """
    Main script for fine-tuning
    """
    print("Loading blueprint context...")
    blueprint_context = load_blueprint_context(BLUEPRINT_CONTEXT_PATH)
    
    print("Loading and processing dataset...")
    dataset = preprocess_dataset(DATASET_PATH, blueprint_context)
    
    # Split dataset for training and evaluation (90/10 split)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    print(f"Training set size: {len(train_dataset)}, Evaluation set size: {len(eval_dataset)}")
    
    print("Loading model and tokenizer...")
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare the model for PEFT/LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Target attention modules
    )
    
    # Get the PEFT model
    model = get_peft_model(model, peft_config)
    
    print("Tokenizing datasets...")
    # Tokenize the datasets
    tokenized_train_dataset = tokenize_dataset(train_dataset, tokenizer)
    tokenized_eval_dataset = tokenize_dataset(eval_dataset, tokenizer)
    
    # Convert tokenized datasets to PyTorch format
    train_dataset = Dataset.from_dict(tokenized_train_dataset)
    eval_dataset = Dataset.from_dict(tokenized_eval_dataset)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="tensorboard",
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print("Fine-tuning complete!")
    
    # Save the final model
    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"Model saved to {OUTPUT_DIR}/final")
    
    # Example of loading and using the fine-tuned model
    print("Example of loading and using the fine-tuned model:")
    print("""
    To use the fine-tuned model:
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    
    # Load the model
    config = PeftConfig.from_pretrained("hvac_finetuned_model/final")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, "hvac_finetuned_model/final")
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    # Function to generate response
    def generate_hvac_response(query, blueprint_context=""):
        formatted_query = f"Building Blueprint Context:\\n{blueprint_context}\\n\\nUser Query: {query}"
        prompt = f"<s>[INST] {formatted_query} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response
    """)

if __name__ == "__main__":
    main()
