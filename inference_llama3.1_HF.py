from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned model checkpoint
model = AutoModelForCausalLM.from_pretrained("./results/checkpoint-10")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

# Ensure the tokenizer has the padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Explicitly set device to GPU 1 (cuda:0 after setting CUDA_VISIBLE_DEVICES=1)
device = torch.device("cuda:0")
model = model.to(device)

# Prepare a new abstract for inference
new_abstract = "This study reveals experimental insights into the transport of amino acids by LAT1 in cancer cells."
new_question = (
    "Does this abstract provide laboratory or experimental evidence that substrate X is transported by transporter protein Y? "
    "Please answer with a confidence score (0-10) and reasoning in a concise format."
)
prompt = f"Abstract: {new_abstract}\nQuestion: {new_question}"

# Tokenize the input and move to the same device as the model
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,  # Adjust max length if needed
).to(device)

# Generate a response
model.eval()
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Explicitly pass the attention mask
        max_length=512,  # Adjust if necessary
        num_return_sequences=1,
        temperature=0.7,  # Adjust temperature for more or less randomness
        top_k=50,        # Control sampling (optional)
        pad_token_id=tokenizer.eos_token_id  # Avoid warnings by explicitly setting pad_token_id
    )

# Decode the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Post-process the response to remove unnecessary or repetitive text
response = response.replace("Best regards, [Your Name]", "").strip()

# Optionally truncate overly verbose reasoning
max_reasoning_length = 1000  # Set a limit for reasoning length
if len(response) > max_reasoning_length:
    response = response[:max_reasoning_length] + "..."

# Print the final response
print("Final response:")
print(response)

# Run with CUDA_VISIBLE_DEVICES=1 python /data/servilla/llama3/inference_llama3.1_HF.py
