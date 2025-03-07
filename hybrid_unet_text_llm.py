import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from diffusers import UNet2DModel, DDPMScheduler
import json
import argparse
from tqdm import tqdm

class HybridTextDiffusion(nn.Module):
    def __init__(self, vocab_size=50257, num_layers=6, seq_length=256, num_timesteps=1000):
        super().__init__()
        self.gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.embed_dim = self.gpt2_config.hidden_size
        self.seq_length = seq_length

        self.token_emb = nn.Embedding(vocab_size, self.embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_length, self.embed_dim))

        self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2_layers = self.gpt2.h[:num_layers]

        self.unet = UNet2DModel(
            sample_size=(1, seq_length),
            in_channels=self.embed_dim,
            out_channels=self.embed_dim,
            layers_per_block=2,
            block_out_channels=(self.embed_dim, self.embed_dim * 2, self.embed_dim * 4),
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D")
        )

        self.to_logits = nn.Linear(self.embed_dim, vocab_size)

        self.timestep_emb = nn.Embedding(num_timesteps, self.embed_dim)

    def forward(self, x, timesteps):
        batch_size = x.size(0)

        x = self.token_emb(x) + self.pos_emb

        t_emb = self.timestep_emb(timesteps).unsqueeze(1)
        x += t_emb

        for layer in self.gpt2_layers:
            x = layer(x)[0]

        x = x.permute(0, 2, 1).unsqueeze(2)

        x = self.unet(x, timesteps).sample

        x = x.squeeze(2).permute(0, 2, 1)
        return self.to_logits(x)

    def print_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total/1e6:.2f}M")


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []

        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc="Processing dataset"):
                text = json.loads(line)['text']
                tokens = self.tokenizer.encode(
                    text,
                    max_length=seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                ).squeeze()
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Initializing tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    model = HybridTextDiffusion(
        vocab_size=tokenizer.vocab_size,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        num_timesteps=args.num_timesteps
    ).to(device)

    model.print_params()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)

    dataset = TextDataset(args.data_path, tokenizer, args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')

        for batch in progress_bar:
            batch = batch.to(device)

            timesteps = torch.randint(0, args.num_timesteps, (batch.size(0),), device=device)

            clean_embeds = model.token_emb(batch)
            noise = torch.randn_like(clean_embeds)
            noisy_embeds = scheduler.add_noise(clean_embeds, noise, timesteps)

            logits = model(batch, timesteps)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch.view(-1),
                ignore_index=tokenizer.pad_token_id
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f'\nEpoch {epoch+1}/{args.epochs} completed. Average Loss: {avg_loss:.4f}')

        if args.save_path:
            checkpoint_path = f"{args.save_path}/model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    print(f"Loading model from {args.model_path}...")
    model = HybridTextDiffusion(
        vocab_size=50257,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        num_timesteps=args.num_timesteps
    ).to(device)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        x = torch.randn(1, args.seq_length, model.embed_dim).to(device)

        print("Starting text generation...")
        for t in reversed(range(args.num_timesteps)):
            timestep = torch.tensor([t], device=device)
            logits = model(x, timestep)
            tokens = torch.argmax(logits, dim=-1)

            if t % 100 == 0:
                print(f'\nStep {t}: {tokenizer.decode(tokens[0])}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'generate'], required=True)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--seq_length', type=int, default=256)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Directory to save model checkpoints')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'generate':
        generate(args)
