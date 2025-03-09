import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
from diffusers import DDPMScheduler
import numpy as np
import argparse
from tqdm import tqdm
import os

class BinaryDataset(Dataset):
    """Loads tokenized data from .bin files"""
    def __init__(self, file_path, seq_length=64):
        self.seq_length = seq_length
        self.data = np.fromfile(file_path, dtype=np.int32).reshape(-1, seq_length)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

class TextDiffusionModel(nn.Module):
    def __init__(
        self, 
        vocab_size=50257,  # GPT-2 vocabulary size
        seq_length=64,
        embed_dim=768,
        num_layers=6,
        num_timesteps=1000,
        num_unet_layers=3
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        # GPT-2 token embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=3072
            ) for _ in range(num_layers)
        ])
        
        # Modified UNet with attention
        self.unet = CodeUNet(embed_dim, seq_length, num_down=num_unet_layers)
        
        # Projection to vocab
        self.to_logits = nn.Linear(embed_dim, vocab_size)
        self.timestep_emb = nn.Embedding(num_timesteps, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, timesteps):
        # Embed tokens
        x = self.token_emb(x) + self.pos_emb[:, :x.size(1), :]
        t_emb = self.timestep_emb(timesteps).unsqueeze(1)
        x += t_emb
        
        # Transformer processing
        for layer in self.transformer_layers:
            x = layer(x)
        
        # UNet processing
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.unet(x, timesteps)
        x = x.squeeze(2).permute(0, 2, 1)
        
        return self.to_logits(self.norm(x))

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model with GPT-2 compatible dimensions
    model = TextDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        seq_length=args.seq_length,
        num_unet_layers=3 if args.seq_length <= 128 else 4
    ).to(device)
    
    # Load binary dataset
    dataset = BinaryDataset(args.data_path, args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch in progress_bar:
            batch = batch.to(device)
            timesteps = torch.randint(0, args.num_timesteps, (batch.size(0),), device=device)
            
            # Diffusion process
            clean_emb = model.token_emb(batch)
            noise = torch.randn_like(clean_emb)
            noisy_emb = scheduler.add_noise(clean_emb, noise, timesteps)
            
            logits = model(noisy_emb, timesteps)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        # Save checkpoint
        torch.save(model.state_dict(), f"{args.save_dir}/checkpoint_epoch_{epoch+1}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seq_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
