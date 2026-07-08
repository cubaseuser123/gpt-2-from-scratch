# GPT-2 From Scratch 🧠

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## What is this?
This is a ground-up, from-scratch implementation of a Generative Pre-trained Transformer (GPT) language model, built entirely in PyTorch. The goal is to recreate the architecture of models like GPT-2 to understand exactly how they work under the hood.

## Inspiration & Intentions
This project is heavily inspired by Andrej Karpathy's "Let's build GPT" tutorials. 

I decided to dive into this project because I wanted to move beyond just using high-level APIs and actually build a strong, first-principles foundation in Deep Learning. By implementing the data pipelines, tokenization, self-attention mechanisms, and transformer blocks myself, my intention is to deeply understand the engineering and mathematical concepts that power modern Large Language Models (LLMs).

## Current Progress 🚧
Currently working through the early stages in `gpt2_from_scratch.ipynb`. So far, the notebook covers:
- Setting up the environment and downloading the `tinyshakespeare` dataset.
- Building a custom character-level tokenizer (encoding strings to integers and decoding back).
- Preparing the dataset by converting it into PyTorch tensors and splitting it into training (90%) and validation (10%) sets.
- Engineering the data loader `get_batch()` function to slice the data into context windows (`block_size`) and stack them into parallel batches (`batch_size`) for GPU processing.
- Implementing a baseline **Bigram Language Model** using `torch.nn.Module`, including an embedding table, cross-entropy loss calculation, and an autoregressive generation loop.
- Setting up the PyTorch `AdamW` optimizer and running a basic training loop to optimize the model and verify token generation.
- Consolidating the core training loop into a master script/cell for clean, repeatable iterations.
- Refactoring the Bigram model to support self-attention by extracting a token embedding layer (`n_embed`), injecting **position embeddings**, and routing through a linear language modeling head.
- Exploring the **mathematical trick in self-attention**, demonstrating how lower-triangular masking (`torch.tril`) and `softmax` allow tokens to selectively aggregate context from previous timesteps using matrix multiplication.

**Next step:** Implementing the actual `Head`, `MultiHeadAttention`, and `FeedForward` PyTorch modules to build a complete Transformer block!
