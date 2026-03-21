import re
import pandas as pd
import json
import os

def split_latex(text):
    # Fixed regex: A-Z and \s (whitespace)
    return re.findall(r"\\[a-zA-Z]+|[^\s]", str(text))

def build_and_save_vocab(csv_path, output_json="vocab.json"):
    print("Building vocabulary...")
    df = pd.read_csv(csv_path)
    
    token_to_id = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    vocab_size = 4
    
    for latex_str in df['label'].dropna():
        tokens = split_latex(latex_str)
        for token in tokens:
            if token not in token_to_id:
                token_to_id[token] = vocab_size
                vocab_size += 1
                
    id_to_token = {v: k for k, v in token_to_id.items()}
    
    vocab_data = {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token
    }
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=4)
        
    print(f"Vocab saved to {output_json} with {vocab_size} tokens!")

if __name__ == "__main__":
    # Run this once from your root SwinMath folder
    build_and_save_vocab("data/train.csv", "vocab.json")