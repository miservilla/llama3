from transformers import AutoModelForCausalLM, TrainingArguments
from transformers import Trainer, DataCollatorForSeq2Seq, AutoProcessor
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import wandb
from dotenv import load_dotenv
import os

# Set W&B API key
wandb.login(os.getenv("WANDB_API_KEY"))

# Load dataset
data_path = "/data/servilla/llama3/data/fine_tuning_data.jsonl"
dataset = load_dataset("json", data_files=data_path)["train"]

# Preprocess function for tokenization
def preprocess_function(examples):
    inputs = [f"Abstract: {abstract}\nQuestion: {question}" for abstract, question in zip(examples["abstract"], examples["question"])]
    outputs = examples["justification"]
    return {
        "input_ids": processor(inputs, truncation=True, padding="max_length", max_length=512)["input_ids"],
        "labels": processor(outputs, truncation=True, padding="max_length", max_length=512)["input_ids"],
    }

# Initialize processor
processor = AutoProcessor.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
if processor.pad_token is None:
    processor.add_special_tokens({'pad_token': '[PAD]'})
    processor.pad_token = processor.eos_token

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Load model with quantization and adapters
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map="auto",
    quantization_config={"load_in_8bit": True}  # New way to specify quantization
)

# Configure LoRA adapters
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Target specific layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Add LoRA adapters to the model
model = get_peft_model(model, lora_config)

# Define data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor,
    model=model,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    run_name="llama3-finetuning",
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=True,
    report_to="wandb",
)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Run with CUDA_VISIBLE_DEVICES=1 python /data/servilla/llama3/fine_tune_llama3.1_HF.py
