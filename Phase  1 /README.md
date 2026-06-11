# Prompt Engineering Playground 🚀

A playground to experiment with, compare, and learn different prompt engineering techniques using Hugging Face's free **Serverless Inference API** (zero API cost, no local GPU/RAM model download required).

## Features

1. **Interactive CLI Chatbot (`chatbot.py`)**: Chat with different Hugging Face models, customize system instructions, adjust temperatures, and observe how the model responds.
2. **Prompt Style Comparison (`prompt_tester.py`)**: Compare different prompting techniques (Zero-Shot, Role Prompting, Chain of Thought, Structured Output) side-by-side.
3. **Interactive Jupyter Notebook (`playground.ipynb`)**: A visual playground to run experiments, test prompt templates, and view formatted comparisons.

---

## Setup Instructions

1. **Virtual Environment**:
   A virtual environment is already created. You can activate it:
   ```bash
   source venv/bin/activate
   ```

2. **Hugging Face Token (Recommended)**:
   * To prevent rate limiting, obtain a free Hugging Face User Access Token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   * Open the `.env` file and add your token:
     ```env
     HF_TOKEN=your_hf_token_here
     ```

---

## How to Use

### 1. Interactive Chatbot
Run the chatbot directly from your terminal:
```bash
python chatbot.py
```
* Custom options will let you choose models, specify system instructions, and control generation settings (temperature, max tokens).

### 2. Prompt Comparison CLI
Run the comparison script to see how different prompt styles perform on a task:
```bash
python prompt_tester.py
```

### 3. Jupyter Notebook Playground
Launch Jupyter Notebook to interactively build and compare prompts:
```bash
jupyter notebook playground.ipynb
```
