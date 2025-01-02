# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List

import fire
import torch
import torch.distributed as dist  # Import torch.distributed

from llama import Llama

# Set default PyTorch settings
torch.set_default_dtype(torch.float32)  # Set default data type for tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Set default device
torch.set_default_device(device)  # Optional: If your PyTorch version supports it
print(f"Default device set to: {device}")  # Informative log


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_gen_len: int = 256,
    max_batch_size: int = 7,
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    # generator = Llama.build(
    #     ckpt_dir=ckpt_dir,
    #     tokenizer_path=tokenizer_path,
    #     max_seq_len=max_seq_len,
    #     max_batch_size=max_batch_size,
    # )

    prompts: List[str] = [
    "What are the primary benefits of renewable energy?",
    "Translate English to Spanish:\n\nwater => agua\ncat => gato\ndog => ",
    "Write a poem about the stars and the night sky.",
    "Write a poem about the bottom of the ocean.",
    "What is the capital city of Australia?",
    "Explain the concept of machine learning in simple terms.",
    "A fast car will",
    "Who is Major Taunt?"
]
    
    # Adjust max_batch_size dynamically based on the number of prompts
    adjusted_batch_size = max(len(prompts), max_batch_size)

    # Pass adjusted_batch_size to Llama.build()
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=adjusted_batch_size,  # Dynamically set batch size
    )


    # prompts: List[str] = [
    #     # For these prompts, the expected answer is the natural continuation of the prompt
    #     "I believe the meaning of life is",
    #     "Simply put, the theory of relativity states that ",
    #     """A brief message congratulating the team on the launch:

    #     Hi everyone,

    #     I just """,
    #     # Few shot prompt (providing a few examples before asking model to complete more);
    #     """Translate English to French:

    #     sea otter => loutre de mer
    #     peppermint => menthe poivrée
    #     plush girafe => girafe peluche
    #     cheese =>""",
    # ]
    
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

    # Destroy the process group
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
