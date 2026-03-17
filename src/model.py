import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class SwinEncoder(nn.Module):
    def __init__(self, d_model = 256):
        super().__init__()
        swin = models.swin_v2_t(weights = models.Swin_T_Weights.DEFAULT)
        self.features = swin.features
        self.project = nn.Linear(768, d_model)
    
    def forward(self, x):
        features = self.features(x)
        sequence = features.flatten(1,2)
        out = self.project(sequence)
        return out
    
class SwinMathModel(nn.Module):
    def __init__(self, vocab_size, d_model = 256, nhead = 8, num_layers = 4, dropout = 0.3):
        super().__init__()
        self.encoder = SwinEncoder(d_model=d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, images, text_targets, tgt_mask = None, tgt_pad_mask = None):
        memory = self.encoder(images)
        memory = self.pos_encoder(memory)
        tgt_emb = self.embedding(text_targets)
        tgt_emb = self.pos_encoder(tgt_emb)
        out = self.decoder(
            tgt = tgt_emb,
            memory = memory,
            tgt_mask = tgt_mask,
            tgt_key_padding_mask = tgt_pad_mask
        )
        return self.fc_out(out)
    
def get_masks(targets, pad_idx, device):
    seq_len = targets.size(1)
    tgt_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    tgt_pad_mask = (targets == pad_idx).to(device)
    return tgt_mask, tgt_pad_mask