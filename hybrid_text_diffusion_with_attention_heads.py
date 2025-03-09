import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from diffusers import DDPMScheduler
import numpy as np
import argparse
from tqdm import tqdm
import os

# ------------------ Core UNet Components ------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,3), padding=(0,1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,3), padding=(0,1)))
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)) if in_channels != out_channels else nn.Identity()
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.gamma * self.conv(x) + self.shortcut(x)

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.SiLU(),
            nn.Linear(dim*4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # [S, B, C]
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.permute(1, 2, 0).unsqueeze(2)  # Restore original shape

class CodeUNet(nn.Module):
    def __init__(self, embed_dim, seq_length, num_down=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_down = num_down

        # Validate sequence length
        assert seq_length % (2**num_down) == 0, \
            f"Seq length {seq_length} must be divisible by 2^{num_down}"

        # Channel progression
        channels = [embed_dim * (2**i) for i in range(num_down + 1)]

        # Downsample path
        self.down_blocks = nn.ModuleList()
        for i in range(num_down):
            self.down_blocks.append(nn.Sequential(
                ResidualBlock(channels[i], channels[i+1]),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
            ))

        # Middle processing
        self.middle = nn.Sequential(
            ResidualBlock(channels[-1], channels[-1]),
            TransformerLayer(channels[-1]),
            ResidualBlock(channels[-1], channels[-1])
        )

        # Upsample path
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_down)):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(channels[i+1], channels[i],
                                 kernel_size=(1,3), stride=(1,2),
                                 padding=(0,1), output_padding=(0,1)),
                ResidualBlock(channels[i], channels[i])
            ))

        self.final = nn.Conv2d(channels[0], embed_dim, kernel_size=(1,1))
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim//4),
            nn.SiLU(),
            nn.Linear(embed_dim//4, embed_dim)

    def forward(self, x, timesteps):
        # Time embedding
        t_emb = self.time_embed(timesteps.float().unsqueeze(-1))
        t_emb = t_emb.view(x.shape[0], self.embed_dim, 1, 1)
        x = x + t_emb

        # Downsample
        skips = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)

        # Middle
        x = self.middle(x)

        # Upsample
        for up in self.up_blocks:
            x = up(x)
            x = torch.cat([x, skips.pop()], dim=1)

        return self.final(x)

# ------------------ Main Model ------------------
class TextDiffusionModel(nn.Module):
    def __init__(self, vocab_size=50257, seq_length=512, embed_dim=768, num_layers=6, num_timesteps=1000):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_length, embed_dim))
        self.time_emb = nn.Embedding(num_timesteps, embed_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, nhead=8)
            for _ in range(num_layers)
        ])

        self.unet = CodeUNet(embed_dim, seq_length, num_down=3)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, timesteps):
        x = self.token_emb(x) + self.pos_emb
        x += self.time_emb(timesteps).unsqueeze(1)

        for layer in self.transformer_layers:
            x = layer(x)

        # UNet processing
        x_unet = x.permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, S]
        x_unet = self.unet(x_unet, timesteps)
        x = x + x_unet.squeeze(2).permute(0, 2, 1)

        return self.head(self.norm(x))

# ------------------ Training System ------------------
class BinaryDataset(Dataset):
    def __init__(self, file_path, seq_length):
        self.data = np.fromfile(file_path, dtype=np.int32).reshape(-1, seq_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = TextDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        seq_length=args.seq_length
    ).to(device)

    dataset = BinaryDataset(args.data_path, args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')

        for batch in progress:
            batch = batch.to(device)
            timesteps = torch.randint(0, args.num_timesteps, (batch.size(0),).to(device)

            # Diffusion process
            clean_emb = model.token_emb(batch)
            noise = torch.randn_like(clean_emb)
            noisy_emb = scheduler.add_noise(clean_emb, noise, timesteps)

            logits = model(noisy_emb, timesteps)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch.view(-1),
                ignore_index=tokenizer.pad_token_id
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        torch.save(model.state_dict(), f"{args.save_dir}/epoch_{epoch+1}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--seq_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
