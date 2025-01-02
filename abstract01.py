from typing import List
import fire
import torch
import torch.distributed as dist
from llama import Llama
import json
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set default PyTorch settings
torch.set_default_dtype(torch.float32)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
logger.info(f"Default device set to: {device}")


def main(
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 512,
    max_batch_size: int = 7,
):
    """
    Main function to generate results for a set of abstracts, checking if 
    they provide laboratory evidence of substrate-protein transport.
    """
    # Initialize distributed processing if environment variables are set
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        logger.info(
            f"Distributed mode initialized: RANK={os.environ['RANK']}, WORLD_SIZE={os.environ['WORLD_SIZE']}")
    else:
        logger.info("Running in non-distributed mode.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"

    # Example dataset of abstracts
    abstracts = [
        "This study investigates the transport of glucose by SGLT1 in human intestinal cells.",
        "The effects of various inhibitors on non-specific diffusion were analyzed.",
        "ATP-sensitive potassium (K (ATP) ) channels are multimeric protein complexes made of four inward rectifying potassium channel (Kir6.x) subunits and four ABC protein sulfonylurea receptor (SURx) subunits. Kir6.x subunits form the potassium ion conducting pore of the channel, and SURx functions to regulate Kir6.x. Kir6.x and SURx are uniquely dependent on each other for expression and function. In pancreatic \u03b2-cells, channels comprising SUR1 and Kir6.2 mediate glucose-stimulated insulin secretion and are the targets of anti-diabetic sulfonylureas. Mutations in genes encoding SUR1 or Kir6.2 are linked to insulin secretion disorders, with loss- or gain-of-function mutations causing congenital hyperinsulinism or neonatal diabetes mellitus, respectively. Defects in the K (ATP) channel in other tissues underlie human diseases of the cardiovascular and nervous systems. Key to understanding how channels are regulated by physiological and pharmacological ligands and how mutations disrupt channel assembly or gating to cause disease is the ability to observe structural changes associated with subunit interactions and ligand binding. While recent advances in the structural method of single-particle cryo-electron microscopy (cryoEM) offers direct visualization of channel structures, success of obtaining high-resolution structures is dependent on highly concentrated, homogeneous K (ATP) channel particles. In this chapter, we describe a method for expressing K (ATP) channels in mammalian cell culture, solubilizing the channel in detergent micelles and purifying K (ATP) channels using an affinity tag to the SURx subunit for cryoEM structural studies."
    ]

    # Create prompts dynamically based on abstracts
    prompts = [
        f"Abstract: {abstract} Question: Does this abstract provide laboratory or experimental evidence that substrate X is transported by transporter protein Y? Provide a confidence score (0-10) where 0 means no evidence and 10 means conclusive evidence. Also, briefly justify your rating."
        for abstract in abstracts
    ]

    # Adjust max_batch_size dynamically
    adjusted_batch_size = min(len(prompts), max_batch_size)

    # Initialize the generator
    generator = Llama.build(
        ckpt_dir='/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct',
        tokenizer_path='/data_link/servilla/.llama/checkpoints/Llama3.1-8B-Instruct/tokenizer.model',
        max_seq_len=max_seq_len,
        max_batch_size=adjusted_batch_size,
        model_parallel_size=1  # Explicitly set for single-GPU mode
    )

    # Log token count for each prompt and check for max_seq_len violations
    for prompt in prompts:
        token_count = len(generator.tokenizer.encode(
            prompt, bos=True, eos=True))
        if token_count > max_seq_len:
            raise ValueError(
                f"Prompt exceeds max_seq_len ({max_seq_len} tokens): {prompt}")
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

        results_list.append({
            "prompt": prompt,
            "confidence_score": confidence_score,
            "justification": justification,
            "response": response
        })

        # Log the results
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Confidence Score: {confidence_score}")
        logger.info(f"Justification: {justification}")
        logger.info("\n==================================\n")

    # Save results to a file
    with open("results.json", "w") as f:
        json.dump(results_list, f, indent=2)

    # Destroy the process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
