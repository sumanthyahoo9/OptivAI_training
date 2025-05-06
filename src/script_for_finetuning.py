"""
Script to fine-tune a Llama model for our explainer
"""
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split

# Configuration
class Config:
    """
    Config for the model and training
    """
    model_name = "/home/sumanthmurthy/llama_models/"  # Path to your local model
    dataset_path = "/home/sumanthmurthy/OptivAI_training/llm_enhanced_training_examples.jsonl"
    output_dir = "/home/sumanthmurthy/llama_finetuned/"
    num_train_epochs = 20
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8
    learning_rate = 2e-4
    max_length = 768
    train_test_split = 0.6  # 60% train, 40% test
    save_steps = 100
    eval_steps = 25
    logging_steps = 10
    quantization = "4bit"  # Options: "4bit", "8bit", or None

# Load and prepare the dataset
def load_and_prepare_dataset(jsonl_path, test_size=0.4):
    """Load JSONL file and split into train/test sets"""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Save splits if desired
    with open(os.path.join(os.path.dirname(Config.dataset_path), 'train_data.jsonl'), 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(os.path.join(os.path.dirname(Config.dataset_path), 'test_data.jsonl'), 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
    
    return train_data, test_data

# Formatting function for the chat template
def format_chat_template(example, tokenizer):
    """Format the example for chat-based fine-tuning"""
    conversation = [
        {"role": "system", "content": "You are an HVAC expert assistant."},
        {"role": "user", "content": example['query']},
        {"role": "assistant", "content": example['response']}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return prompt

# Tokenization function
def tokenize_function(examples, tokenizer):
    """Tokenize the formatted prompts"""
    # For batched processing
    if isinstance(examples, dict) and isinstance(examples.get('query', []), list):
        batch_size = len(examples['query'])
        formatted_texts = []
        
        for i in range(batch_size):
            # Manual formatting instead of using chat template
            example = {key: examples[key][i] for key in examples if i < len(examples[key])}
            # Simple text format without using apply_chat_template
            text = f"<s>[INST] <<SYS>>\nYou are an HVAC expert assistant.\n<</SYS>>\n\n{example['query']} [/INST] {example['response']}</s>"
            formatted_texts.append(text)
    else:
        # Handle single example case
        text = f"<s>[INST] <<SYS>>\nYou are an HVAC expert assistant.\n<</SYS>>\n\n{examples['query']} [/INST] {examples['response']}</s>"
        formatted_texts = [text]
    
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        padding="max_length",
        max_length=Config.max_length,
        return_tensors="pt"
    )
    
    # For causal LM, the labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

# The main fine-tune function
def finetune_llama():
    """
    Fine-tune the Llama model
    """
    # Set up the quantization configuration
    if Config.quantization == "4bit":
        print("Using 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    elif Config.quantization == "8bit":
        print("Using 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        print("Using fp16 precision...")
        bnb_config = None
    
    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    # Add padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Prepare model for the fine-tuning
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    print("Loading dataset...")
    train_data, test_data = load_and_prepare_dataset(Config.dataset_path, test_size=(1-Config.train_test_split))

    # Create HF datasets
    train_dataset = load_dataset('json', data_files=os.path.join(os.path.dirname(Config.dataset_path), 'train_data.jsonl'), split='train')
    test_dataset = load_dataset('json', data_files=os.path.join(os.path.dirname(Config.dataset_path), 'test_data.jsonl'), split='train')

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    tokenized_test = test_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=test_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="max_length",
        max_length=Config.max_length,
    )
    
    # Create output dir if it doesn't exist
    os.makedirs(Config.output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        num_train_epochs=Config.num_train_epochs,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_train_batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        eval_steps=Config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=0.3,
        save_total_limit=3,
        push_to_hub=False,
        report_to="tensorboard",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(Config.output_dir)
    tokenizer.save_pretrained(Config.output_dir)
    
    print(f"Model saved to {Config.output_dir}")

if __name__ == "__main__":
    finetune_llama()