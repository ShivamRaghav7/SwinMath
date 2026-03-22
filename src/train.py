import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset import HMEDataset, ResizeAndPadSquare, CollateFn
from tokenizer import MathTokenizer
from model import SwinMathModel, get_masks

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
MAX_LEN = 150 
IMG_DIR = "data/images"        
TRAIN_CSV = "data/train.csv"   
VAL_CSV = "data/val.csv"       
CHECKPOINT_DIR = "checkpoints" 
METRICS_FILE = "metrics.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    tokenizer = MathTokenizer("vocab.json")
    transform = ResizeAndPadSquare(target_size=256)
    pad_idx = tokenizer.token_to_id[tokenizer.PAD]

    train_dataset = HMEDataset(TRAIN_CSV, IMG_DIR, tokenizer, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=CollateFn(pad_idx),
        num_workers=2 
    )

    val_dataset = HMEDataset(VAL_CSV, IMG_DIR, tokenizer, transform=transform)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=CollateFn(pad_idx),
        num_workers=2
    )

    model = SwinMathModel(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,val_accuracy\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, leave=True, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
        for images, texts in loop:
            images, texts = images.to(device), texts.to(device)
            decoder_input = texts[:, :-1]
            expected_output = texts[:, 1:]
            tgt_mask, tgt_pad_mask = get_masks(decoder_input, pad_idx, device)
            optimizer.zero_grad()
            logits = model(images, decoder_input, tgt_mask, tgt_pad_mask)

            # Reshape for Loss
            logits_reshaped = logits.reshape(-1, logits.shape[-1])
            expected_output_reshaped = expected_output.reshape(-1)
            loss = criterion(logits_reshaped, expected_output_reshaped)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct_tokens = 0
        total_tokens = 0

        val_loop = tqdm(val_loader, leave=True, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ")
        with torch.no_grad(): 
            for images, texts in val_loop:
                images, texts = images.to(device), texts.to(device)
                
                decoder_input = texts[:, :-1]
                expected_output = texts[:, 1:]
                tgt_mask, tgt_pad_mask = get_masks(decoder_input, pad_idx, device)

                logits = model(images, decoder_input, tgt_mask, tgt_pad_mask)
                
                logits_reshaped = logits.reshape(-1, logits.shape[-1])
                expected_output_reshaped = expected_output.reshape(-1)
                
                loss = criterion(logits_reshaped, expected_output_reshaped)
                val_loss += loss.item()

                # Calculate Accuracy (ignoring <PAD> tokens)
                predictions = torch.argmax(logits_reshaped, dim=1)
                mask = expected_output_reshaped != pad_idx
                
                correct_tokens += (predictions[mask] == expected_output_reshaped[mask]).sum().item()
                total_tokens += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_tokens / total_tokens

        print(f"\n--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}\n")

        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_accuracy:.4f}\n")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"swin_math_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)

if __name__ == "__main__":
    main()