import json
import torch
from architecture import Transformer, ModelArgs  # Your model architecture
from tokenizer import Tokenizer  # Your tokenizer implementation
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, destroy_model_parallel
import torch.distributed as dist
import os

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"  # Use an available port


if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=1, rank=0)


# Paths
TOKENIZER_PATH = "/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model"
MODEL_PATH = "/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct/consolidated.00.pth"
DATA_PATH = "/data_link/servilla/llama3/data/data.json"
OUTPUT_PATH = "/data_link/servilla/llama3/fine_tuned_llama3.pth"

# Hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
EPOCHS = 3

# Initialize device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model parallelism for a single GPU
initialize_model_parallel(model_parallel_size_=1)


try:
    print("Loading tokenizer...")
    tokenizer = Tokenizer(TOKENIZER_PATH)
    print("Tokenizer loaded successfully.")

    print("Initializing model...")
    args = ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=tokenizer.model.n_vocab,  # Corrected to use n_vocab
        multiple_of=256,
        max_batch_size=BATCH_SIZE,
        max_seq_len=2048,
    )
    model = Transformer(args).to(DEVICE)
    print("Model initialized successfully.")

    # Load pre-trained weights
    print("Loading model weights...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)
    print("Model weights loaded successfully.")

    # Load dataset
    print("Loading dataset...")
    with open(DATA_PATH, "r") as f:
        dataset = json.load(f)
    print("Dataset loaded successfully.")

    # Prepare optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for i, example in enumerate(dataset):
            input_text = (
                f"Abstract: {example['abstract']} "
                f"Question: {example['question']}"
            )
            target_text = (
                f"Confidence Score: {example['confidence_score']} "
                f"Justification: {example['justification']}"
            )

            # Tokenize inputs and targets
            input_ids = torch.tensor(
                tokenizer.encode(input_text, bos=True, eos=True), device=DEVICE
            ).unsqueeze(0)
            target_ids = torch.tensor(
                tokenizer.encode(target_text, bos=True, eos=True), device=DEVICE
            ).unsqueeze(0)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, start_pos=0)
            logits = outputs[:, :-1, :].contiguous()
            labels = target_ids[:, 1:].contiguous()

            loss = torch.nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Logging
            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataset)}], Loss: {loss.item():.4f}"
                )

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] completed. Total Loss: {total_loss:.4f}")

    # Save fine-tuned model
    print(f"Saving fine-tuned model to {OUTPUT_PATH}...")
    torch.save(model.state_dict(), OUTPUT_PATH)
    print("Fine-tuned model saved successfully.")

finally:
    # Clean up model parallel group
    destroy_model_parallel()
    print("Model parallelism destroyed.")
