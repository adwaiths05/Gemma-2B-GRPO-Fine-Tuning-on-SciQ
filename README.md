# Gemma-2B GRPO Fine-Tuning on SciQ

This repository contains code to fine-tune Google’s Gemma-2B model on the SciQ dataset using GRPO (Generative Reward Policy Optimization) for reasoning tasks.

> **⚠️ Important:**  

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

---

## Overview

Gemma-2B is a large language model developed by Google. This project demonstrates how to:

- Load SciQ, a science QA dataset.
- Format multiple-choice questions as prompts.
- Define a reward function based on correct answers.
- Fine-tune Gemma-2B using GRPO to improve reasoning.

All fine-tuning is done for research and experimentation.

---

## Features

- Tokenization and prompt formatting for SciQ.
- Dynamic reward computation for GRPO.
- Code ready to run on Kaggle with 4-bit quantization for GPU efficiency.

---

## Requirements

- Python >= 3.9
- PyTorch >= 2.x
- Hugging Face Transformers
- `trl==0.4.1` (important for GRPO compatibility)
- Datasets library

**Install dependencies:**
```sh
pip install torch transformers datasets trl==0.4.1
```

---

## Setup

1. **Request access to Gemma-2B on Hugging Face.**

2. **Generate a Hugging Face token and log in in your notebook:**
   ```python
   from huggingface_hub import notebook_login
   notebook_login()  # paste your HF token
   ```

3. **Clone this repository or download the notebook files.**

4. **Run the notebook on Kaggle or your local environment.**

---

## Usage

1. **Load the SciQ dataset and format prompts.**
2. **Define the reward function:**
   ```python
   def reward_fn(prompts, completions, references):
       rewards = []
       for completion, ref in zip(completions, references):
           rewards.append(1.0 if ref in completion.upper() else 0.0)
       return rewards
   ```

3. **Initialize the GRPO trainer (`trl==0.4.1`):**
   ```python
   trainer = GRPOTrainer(
       model=model,
       train_dataset=train_dataset,
       reward_function=reward_fn,
       config=config
   )
   ```

4. **Train and save your fine-tuned model:**
   ```python
   trainer.train()
   trainer.save_model("/kaggle/working/gemma2b-sciq-grpo")
   ```


---

## License

This project does **not** distribute Gemma-2B. You must accept the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) before using the model.

The repository code itself is available under the [MIT License](LICENSE).
