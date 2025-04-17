import time
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

start_time = time.time()  # Start timing

# 1. Load toy dataset
print("Loading dataset...")
dataset = load_from_disk("./toy_en_de")
print(f"Dataset loaded with {len(dataset)} examples")

# Use a small pre-trained model for fine-tuning
# Instead of a model requiring sentencepiece, let's use BART which doesn't have that dependency
model_name = "facebook/bart-base"  # Using BART which has a similar architecture
print(f"Using model: {model_name}")

# 2. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Note: BART doesn't need src_lang and tgt_lang to be set

# Preprocess function for tokenization
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["de"] for ex in examples["translation"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        padding="longest",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    
    # Tokenize targets
    # BART requires a different approach for tokenizing targets
    labels = tokenizer(
        targets,
        padding="longest",
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing to the dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
)
print("Tokenization completed")

# 3. Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./ckpt",
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    num_train_epochs=1,
    save_total_limit=1,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    max_steps=750,  # ~10 min on mid-tier GPU
    save_strategy="steps",
    save_steps=250,
    label_smoothing_factor=0.1,  # Moved from trainer to training args
    report_to="none",  # Disable Tensorboard and other reporters
)

# 4. Initialize trainer
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 5. Train and save checkpoint
print("Starting training...")
trainer.train()
print("Training completed")

# Save the final model
trainer.save_model()
print("Model saved to ./ckpt")

# 6. Print total runtime
end_time = time.time()
runtime_seconds = int(end_time - start_time)
print(f"Total runtime: {runtime_seconds} seconds ({runtime_seconds // 60}m {runtime_seconds % 60}s)")
