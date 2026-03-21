import torch
from PIL import Image
import os

# Import our custom architecture and tools
from model import SwinMathModel
from tokenizer import MathTokenizer
from dataset import ResizeAndPadSquare

# --- CONFIG ---
# Make sure to update this to the exact name of your saved weights!
CHECKPOINT_PATH = "checkpoints/swin_math_epoch_10.pth" 
VOCAB_PATH = "vocab.json"
MAX_LEN = 150

def greedy_decode(model, image_tensor, tokenizer, device):
    """Generates LaTeX token-by-token using the trained model."""
    model.eval()
    
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    sos_idx = tokenizer.token_to_id[tokenizer.SOS]
    eos_idx = tokenizer.token_to_id[tokenizer.EOS]
    pad_idx = tokenizer.token_to_id[tokenizer.PAD]
    
    # Start the sequence with just the <SOS> token
    generated_tokens = torch.tensor([[sos_idx]], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # Step 1: Extract visual features ONLY ONCE to save massive compute time
        memory = model.encoder(image_tensor)
        memory = model.pos_encoder(memory)
        
        # Step 2: The Autoregressive Loop
        for _ in range(MAX_LEN):
            seq_len = generated_tokens.size(1)
            
            # Create masks for the current sequence
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            tgt_pad_mask = (generated_tokens == pad_idx).to(device)

            # Pass current sequence through decoder
            tgt_emb = model.embedding(generated_tokens)
            tgt_emb = model.pos_encoder(tgt_emb)
            
            out = model.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask
            )
            
            logits = model.fc_out(out)
            
            # Grab the model's prediction for the NEXT token (the last one in the sequence)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # Append the new token to our running list
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            
            # If the model says it is done, break the loop
            if next_token.item() == eos_idx:
                break
                
    # Step 3: Clean up the output
    token_ids = generated_tokens.squeeze().tolist()
    
    # Strip off the <SOS> and <EOS> tokens before converting to text
    if token_ids[0] == sos_idx:
        token_ids = token_ids[1:]
    if token_ids and token_ids[-1] == eos_idx:
        token_ids = token_ids[:-1]
        
    return tokenizer.decode(token_ids)

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")
    
    # Load Tools
    tokenizer = MathTokenizer(VOCAB_PATH)
    model = SwinMathModel(vocab_size=tokenizer.vocab_size).to(device)
    
    # Load the trained brain (weights_only=True is a modern PyTorch security standard)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Could not find weights at {CHECKPOINT_PATH}. Did you train the model yet?")
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    
    # Process the Image
    img = Image.open(image_path).convert('RGB')
    transform = ResizeAndPadSquare(target_size=256)
    img_tensor = transform(img)
    
    # Get the LaTeX!
    latex_prediction = greedy_decode(model, img_tensor, tokenizer, device)
    
    print("\n" + "="*50)
    print(f"IMAGE: {image_path}")
    print(f"LATEX PREDICTION:\n{latex_prediction}")
    print("="*50 + "\n")
    
    return latex_prediction

if __name__ == "__main__":
    # Test it by pointing it at a single image in your dataset
    # Change 'test_math_1.jpg' to an actual filename in your folder
    predict_image("data/images/test_math_1.jpg")