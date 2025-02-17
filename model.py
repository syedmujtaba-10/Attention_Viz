from transformers import AutoModel, AutoTokenizer
import torch
from collections import Counter

# Available models for selection
MODEL_OPTIONS = {
    "BERT Base Uncased": "bert-base-uncased",
    "BERT Large Uncased": "bert-large-uncased",
    "DistilBERT Base": "distilbert-base-uncased",
    "GPT-2 Small": "gpt2",
    "GPT-2 Medium": "gpt2-medium",
    "RoBERTa Base": "roberta-base",
    "RoBERTa Large": "roberta-large"
}

def load_model(model_name):
    """Loads the tokenizer and model based on user selection."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

def make_tokens_unique(tokens):
    """Appends a count to duplicate tokens to ensure uniqueness."""
    counts = Counter()
    unique_tokens = []
    for token in tokens:
        if counts[token] > 0:
            unique_tokens.append(f"{token}_{counts[token]}")  # Add index to duplicates
        else:
            unique_tokens.append(token)
        counts[token] += 1
    return unique_tokens

def get_attention(text, model_name):
    """Loads model dynamically, tokenizes input text, and returns attention weights."""
    tokenizer, model = load_model(model_name)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    # Extract original tokens and make them unique
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
    unique_tokens = make_tokens_unique(tokens)

    return outputs.attentions, unique_tokens
