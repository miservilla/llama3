from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

from transformers import LlamaTokenizer

# Load dataset
data = Dataset.from_json("/data_link/servilla/llama3/data/data.json")

# Add prompt formatting


def preprocess_function(example):
    input_text = f"Abstract: {example['abstract']} Question: {example['question']}"
    label_text = f"Confidence Score: {example['confidence_score']} Justification: {example['justification']}"
    return {"input_text": input_text, "label_text": label_text}


data = data.map(preprocess_function)

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(
    "/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct", legacy=False)
print(tokenizer)
tokenizer.add_special_tokens({"additional_special_tokens": [
                             "Abstract:", "Question:", "Confidence Score:", "Justification:"]})
model = LlamaForCausalLM.from_pretrained(
    "/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct")
model.resize_token_embeddings(len(tokenizer))

# Tokenize data


def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"], max_length=512, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        examples["label_text"], max_length=512, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_data = data.map(tokenize_function, batched=True)

# Split into train and eval
tokenized_data = tokenized_data.train_test_split(test_size=0.2)

# Set up Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    fp16=True,  # Enable mixed precision
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("fine_tuned_llama3_model")
