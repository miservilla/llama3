import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
import json

# Paths
MODEL_PATH = "/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct/consolidated.00.pth"
TOKENIZER_PATH = "/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model"
DATASET_PATH = "/data_link/servilla/llama3/data/data.json"

# Load SentencePiece tokenizer
print("Loading tokenizer...")
tokenizer = spm.SentencePieceProcessor()
if not tokenizer.Load(TOKENIZER_PATH):
    raise RuntimeError(f"Failed to load tokenizer model from {TOKENIZER_PATH}")

# Load the model
print("Loading model...")
model = torch.load(MODEL_PATH, map_location="cuda")
model = model.to("cuda")
print("Model loaded successfully.")

# Define LoRA layer


class LoRALayer(nn.Module):
    def __init__(self, original_dim, lora_r, lora_alpha=1.0, lora_dropout=0.1):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(original_dim, lora_r))
        self.lora_B = nn.Parameter(torch.zeros(lora_r, original_dim))
        self.scaling = lora_alpha / lora_r
        self.dropout = nn.Dropout(p=lora_dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        lora_out = self.dropout(
            x @ self.lora_A @ self.lora_B.t()) * self.scaling
        return x + lora_out


# Integrate LoRA into the model
for name, module in model.named_modules():
    if "attention" in name:  # Replace with your model's attention layer naming
        module.attention_weights = LoRALayer(
            original_dim=module.attention_weights.size(-1),
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.1,
        )

# Freeze base parameters
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

# Dataset and Dataloader


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_text = f"Abstract: {example['abstract']} Question: {example['question']}"
        label_text = f"Confidence Score: {example['confidence_score']} Justification: {example['justification']}"
        input_ids = self.tokenizer.EncodeAsIds(input_text)[: self.max_length]
        label_ids = self.tokenizer.EncodeAsIds(label_text)[: self.max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
        }


# Load and preprocess dataset
print("Loading dataset...")
with open(DATASET_PATH, "r") as f:
    raw_data = json.load(f)

train_size = int(0.8 * len(raw_data))
train_data = raw_data[:train_size]
val_data = raw_data[train_size:]

train_dataset = CustomDataset(train_data, tokenizer)
val_dataset = CustomDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

# Training loop


def train(model, train_loader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to("cuda")
            label_ids = batch["label_ids"].to("cuda")
            outputs = model(input_ids)
            logits = outputs["logits"]
            loss = criterion(logits.view(-1, logits.size(-1)),
                             label_ids.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluation loop


def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to("cuda")
            label_ids = batch["label_ids"].to("cuda")
            outputs = model(input_ids)
            logits = outputs["logits"]
            loss = criterion(logits.view(-1, logits.size(-1)),
                             label_ids.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")


# Fine-tune the model
print("Starting training...")
train(model, train_loader, optimizer, criterion)
print("Training completed.")

# Evaluate the model
print("Starting evaluation...")
evaluate(model, val_loader, criterion)

# Save the fine-tuned model
print("Saving the fine-tuned model...")
torch.save(model.state_dict(), "./fine_tuned_llama3_8b_lora.pth")
print("Model saved.")
