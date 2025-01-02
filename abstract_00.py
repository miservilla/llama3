from typing import List
import fire
import torch
import torch.distributed as dist
from llama import Llama
import json
import re
import os

# Set default PyTorch settings
torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
print(f"Default device set to: {device}")


def main(
    # ckpt_dir: str,
    # tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 512,
    max_batch_size: int = 7,
):
    # Initialize distributed processing if environment variables are set
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        print(
            f"Distributed mode initialized: RANK={os.environ['RANK']}, WORLD_SIZE={os.environ['WORLD_SIZE']}")
    else:
        print("Running in non-distributed mode.")

    # Example dataset
    abstracts = [
        "This study investigates the transport of glucose by SGLT1 in human intestinal cells.",
        "The effects of various inhibitors on non-specific diffusion were analyzed.",
    ]

    # Create prompts dynamically
    prompts = [
        f"Abstract: {abstract} Question: Does this abstract provide laboratory or experimental evidence that substrate X is transported by transporter protein Y? Provide a confidence score (0-10) where 0 means no evidence and 10 means conclusive evidence. Also, briefly justify your rating."
        for abstract in abstracts
    ]

    # Adjust max_batch_size dynamically
    adjusted_batch_size = max(len(prompts), max_batch_size)

    # Initialize the generator
    generator = Llama.build(
        ckpt_dir='/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct',
        tokenizer_path='/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model',
        max_seq_len=max_seq_len,
        max_batch_size=adjusted_batch_size,
        model_parallel_size=1  # Explicitly set to single-GPU
    )

    # Log the token count for each prompt
    for prompt in prompts:
        token_count = len(generator.tokenizer.encode(
            prompt, bos=True, eos=True))
        print(f"Token count for prompt: {token_count}")

    # Clean up the distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()

    # Generate results
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Process and save results
    results_list = []
    # for prompt, result in zip(prompts, results):
    #     response = result['generation']
    #     results_list.append({"prompt": prompt, "response": response})
    #     print(f"Prompt: {prompt}")
    #     print(f"Response: {response}")
    #     print("\n==================================\n")
    # Extract and process outputs
    for prompt, result in zip(prompts, results):
        response = result['generation']

        # Extract confidence score
        score_match = re.search(
            r"(confidence score|rating|confidence level)[:=]?\s*(\d+)",
            response,
            re.IGNORECASE,
        )
        confidence_score = score_match.group(2) if score_match else "N/A"

        # Extract justification
        justification_match = re.search(
            r"Justification: (.*?)(?:Answer:|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        justification = justification_match.group(1).strip(
        ) if justification_match else "No justification provided."

        print(f"Prompt: {prompt}")
        print(f"Confidence Score: {confidence_score}")
        print(f"Justification: {justification}")
        print("\n==================================\n")

    # Save results to a file
    with open("results.json", "w") as f:
        json.dump(results_list, f, indent=2)

    # Destroy the process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
