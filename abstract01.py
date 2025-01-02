from typing import List
import fire
import torch
import torch.distributed as dist
from llama import Llama
import json
import re
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set default PyTorch settings
torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
logger.info(f"Default device set to: {device}")


def main(
    temperature: float = 0.15,  # Slightly lower temperature
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 256,
    max_batch_size: int = 7,
):
    # Initialize distributed processing if environment variables are set
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        logger.info(
            f"Distributed mode initialized: RANK={os.environ['RANK']}, WORLD_SIZE={os.environ['WORLD_SIZE']}")
    else:
        logger.info("Running in non-distributed mode.")

    # Example dataset
    abstracts = [
        "This study investigates the transport of glucose by SGLT1 in human intestinal cells.",
        "The effects of various inhibitors on non-specific diffusion were analyzed.",
    ]

    # Create prompts dynamically
    prompts = [
        f"Abstract: {abstract}\n\n"
        "Question: Does this abstract provide laboratory or experimental evidence supporting the transport of substrate X by protein Y?\n"
        "Respond in the following format:\n"
        "- Confidence Score (0-10): [Your score]\n"
        "- Justification: [Concise justification based on the abstract]"
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
        logger.info(f"Token count for prompt: {token_count}")

    # Generate results
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Process and save results
    results_list = []
    for prompt, result in zip(prompts, results):
        response = result['generation']

        # Extract confidence score
        score_match = re.search(
            r"(confidence score|rating|score|evaluation)[:=\s]*([0-9]+(?:\.[0-9]+)?)",
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
        # Remove repetitive patterns
        justification = re.sub(r"(0\.)+", "0", justification)
        # Collapse excessive spaces
        justification = re.sub(r"\s+", " ", justification)

        logger.info(f"Prompt: {prompt}")
        logger.info(f"Confidence Score: {confidence_score}")
        logger.info(f"Justification: {justification}")
        logger.info("\n==================================\n")

        results_list.append(
            {"prompt": prompt, "confidence_score": confidence_score, "justification": justification})

    # Save results to a file
    with open("results.json", "w") as f:
        json.dump(results_list, f, indent=2)

    # Clean up the distributed process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
