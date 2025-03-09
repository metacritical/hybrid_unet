import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
from diffusers import DDPMScheduler
import json
import argparse
from tqdm import tqdm
import os
import numpy as np
from torch.utils.checkpoint import checkpoint

class SelfAttention2D(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        return attn_output

class CodeUNet(nn.Module):
    def __init__(self, embed_dim, seq_length, num_down_steps=3, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_down_steps = num_down_steps
        if seq_length % (2 ** num_down_steps) != 0:
            raise ValueError(f"Sequence length {seq_length} must be divisible by 2^{num_down_steps}")

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.seq_lengths = [seq_length // (2 ** i) for i in range(num_down_steps + 1)]
        self.channels = [embed_dim * (2 ** i) for i in range(num_down_steps + 1)]

        # Down blocks
        for i in range(num_down_steps):
            in_ch = self.channels[i]
            out_ch = self.channels[i+1]
            seq_len = self.seq_lengths[i]
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(1,3), padding=(0,1)),
                nn.LayerNorm([out_ch, 1, seq_len]),
                nn.SiLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=(1,3), padding=(0,1)),
                nn.LayerNorm([out_ch, 1, seq_len]),
                nn.SiLU(),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
            )
            self.down_blocks.append(block)

        # Middle block with attention
        mid_ch = self.channels[-1]
        mid_seq_len = self.seq_lengths[-1]
        self.middle = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(1,3), padding=(0,2), dilation=(1,2)),
            nn.LayerNorm([mid_ch, 1, mid_seq_len]),
            nn.SiLU(),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=(1,3), padding=(0,1)),
            nn.LayerNorm([mid_ch, 1, mid_seq_len]),
            SelfAttention2D(mid_ch, num_heads),
            nn.SiLU()
        )

        # Up blocks
        for i in range(num_down_steps):
            in_ch = self.channels[num_down_steps - i]
            out_ch = self.channels[num_down_steps - i - 1]
            seq_len = self.seq_lengths[num_down_steps - i - 1]
            block = nn.Sequential(
                nn.ConvTranspose2d(in_ch * 2, out_ch, kernel_size=(1,3), stride=(1,2),
                                  padding=(0,1), output_padding=(0,1)),
                nn.LayerNorm([out_ch, 1, seq_len]),
                nn.SiLU()
            )
            self.up_blocks.append(block)

        self.final = nn.Conv2d(embed_dim, embed_dim, kernel_size=(1,1))
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim//4),
            nn.SiLU(),
            nn.Linear(embed_dim//4, embed_dim)
        )

    def forward(self, x, timesteps):
        skips = []
        t_emb = self.time_embed(timesteps.view(-1, 1).float()).view(x.size(0), self.embed_dim, 1, 1)
        x += t_emb

        # Downsample
        for block in self.down_blocks:
            x = checkpoint(block, x)
            skips.append(x)

        # Middle
        x = self.middle(x)

        # Upsample with skip connections
        for i, block in enumerate(self.up_blocks):
            x = torch.cat([x, skips.pop()], dim=1)
            x = checkpoint(block, x)

        return self.final(x)

class TextDiffusionModel(nn.Module):
    def __init__(self, vocab_size, seq_length, num_layers=6, num_timesteps=1000, num_down_steps=3):
        super().__init__()
        self.config = GPT2Config.from_pretrained('gpt2')
        self.embed_dim = self.config.n_embd
        self.seq_length = seq_length

        self.token_emb = nn.Embedding(vocab_size, self.embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_length, self.embed_dim))
        self.text_model = GPT2Model.from_pretrained('gpt2')
        self.text_layers = self.text_model.h[:num_layers]
        self.unet = CodeUNet(self.embed_dim, seq_length, num_down_steps=num_down_steps)
        self.to_logits = nn.Linear(self.embed_dim, vocab_size)
        self.timestep_emb = nn.Embedding(num_timesteps, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward_tokens(self, tokens, timesteps):
        """Process token indices through the model"""
        batch_size = tokens.size(0)
        x = self.token_emb(tokens) + self.pos_emb[:, :tokens.size(1), :]
        t_emb = self.timestep_emb(timesteps).unsqueeze(1)
        x += t_emb
        for layer in self.text_layers:
            x = layer(x)[0]
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.unet(x, timesteps)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.norm(x)
        return self.to_logits(x)

    def forward_embeddings(self, embeddings, timesteps):
        """Process embeddings directly (for noised embeddings)"""
        batch_size = embeddings.size(0)

        # Add timestep embedding
        t_emb = self.timestep_emb(timesteps).unsqueeze(1)
        x = embeddings + t_emb

        # RoBERTa processing
        for layer in self.text_layers:
            x = layer(x)[0]

        # UNet requires [batch, channels, height, width]
        x = x.permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, S]

        # UNet processing with timestep conditioning
        x = self.unet(x, timesteps)

        # Reshape back to sequence
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.norm(x)
        return self.to_logits(x)

    def print_params(self):
        """Print model parameter count for debugging"""
        total = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total/1e6:.2f}M")


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc="Processing dataset"):
                try:
                    parsed = json.loads(line)
                    text = parsed.get('text', parsed.get('content', parsed.get('body', '')))
                except json.JSONDecodeError:
                    try:
                        arr = np.fromstring(line.strip(), dtype=int, sep=' ')
                        self.data.append(torch.tensor(arr, dtype=torch.long))
                        continue
                    except:
                        text = line
                # In the TextDataset __init__ method
                tokens = self.tokenizer.encode(
                    text,
                    max_length=seq_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                    # Add these parameters to handle padding explicitly
                    pad_to_max_length=True,
                    padding_side='right'
                ).squeeze()
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize tokenizer with padding token
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Required for GPT-2

    # Create model with optimized configuration
    model = TextDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        seq_length=args.seq_length,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        num_down_steps=args.num_down_steps
    ).to(device)

    # Explicitly set padding index for embedding layer
    model.token_emb.padding_idx = tokenizer.pad_token_id

    # Print parameter count for verification
    model.print_params()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        fused=True  # Enable fused optimizer for speed
    )

    scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Dataset and dataloader
    dataset = TextDataset(args.data_path, tokenizer, args.seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,  # Optimize memory transfer
        num_workers=4     # Parallel data loading
    )

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')

        for batch in progress_bar:
            batch = batch.to(device, non_blocking=True)
            timesteps = torch.randint(
                0, args.num_timesteps,
                (batch.size(0),),
                device=device
            )

            # Mixed precision context
            with torch.amp.autocast('cuda'):
                # Get clean embeddings
                with torch.no_grad():
                    clean_embeddings = model.token_emb(batch)

                # Add noise
                noise = torch.randn_like(clean_embeddings)
                noisy_embeddings = scheduler.add_noise(clean_embeddings, noise, timesteps)

                # Forward pass
                logits = model.forward_embeddings(noisy_embeddings, timesteps)

                # Calculate loss
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch.reshape(-1),
                    ignore_index=tokenizer.pad_token_id
                )

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update progress
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Epoch completion
        avg_loss = total_loss / len(dataloader)
        print(f'\nEpoch {epoch+1}/{args.epochs} completed. Average Loss: {avg_loss:.4f}')

        # Save checkpoint
        checkpoint_path = f"{args.save_path}/codediffusion_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


def low_confidence_remasking(probs, num_to_mask):
    """
    Apply the low-confidence remasking strategy from the LLaDA paper.
    Remasks tokens with the lowest confidence predictions.
    """
    confidences, _ = torch.max(probs, dim=-1)
    # Find the least confident tokens
    _, indices = torch.topk(confidences, k=num_to_mask, largest=False)
    return indices


def generate(args):
    """
    Generate text using diffusion sampling strategies
    implemented from the LLaDA paper insights
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from {args.model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Fix: Use GPT-2 instead of BERT
    tokenizer.pad_token = tokenizer.eos_token         # Required for GPT-2
    model = TextDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        seq_length=args.seq_length,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        num_down_steps=args.num_down_steps  # Add missing parameter
    ).to(device)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Prompt processing
    if args.prompt:
        prompt_tokens = tokenizer.encode(args.prompt, return_tensors='pt').to(device)
        prompt_len = prompt_tokens.size(1)
    else:
        prompt_tokens = None
        prompt_len = 0

    # Start with a fully masked sequence for generation
    x = torch.full((1, args.gen_length), tokenizer.mask_token_id, dtype=torch.long).to(device)

    # If prompt provided, include it at the beginning
    if prompt_tokens is not None:
        x[0, :prompt_len] = prompt_tokens[0]

    print("Starting code generation with diffusion process...")
    with torch.no_grad():
        # Reverse diffusion process from t=1 to t=0
        for t in tqdm(range(args.num_timesteps-1, -1, -args.sampling_stride), desc="Generating"):
            # Current timestep
            timestep = torch.tensor([t], device=device)

            # Forward pass to get logits
            logits = model.forward_tokens(x, timestep)
            probs = torch.softmax(logits, dim=-1)

            # Update non-masked tokens (they stay the same)
            mask = (x != tokenizer.mask_token_id)

            # For masked tokens, predict and potentially remask some
            if t > 0:  # Not the final step
                pred_tokens = torch.argmax(probs, dim=-1)

                # Create a new mask for the next step, using either:
                # 1. Low confidence remasking strategy (from LLaDA paper)

                # Calculate how many tokens should still be masked at the next step
                next_t = max(0, t - args.sampling_stride)
                current_ratio = float(t) / args.num_timesteps
                next_ratio = float(next_t) / args.num_timesteps

                # Get indices of tokens to remask
                filled_indices = torch.where(~mask)[1]
                if len(filled_indices) > 0:
                    num_to_remask = int(len(filled_indices) * next_ratio / current_ratio)
                    if num_to_remask > 0:
                        # Get confidences only for the currently masked (filled) positions
                        masked_probs = probs[0, filled_indices]
                        remask_local_indices = low_confidence_remasking(masked_probs, num_to_remask)
                        remask_global_indices = filled_indices[remask_local_indices]

                        # Create a new sequence with some tokens remasked
                        new_x = pred_tokens.clone()
                        new_x[0, remask_global_indices] = tokenizer.mask_token_id
                        x = new_x
                    else:
                        x = pred_tokens
                else:
                    x = pred_tokens
            else:
                # Final step - no more remasking
                x = torch.argmax(probs, dim=-1)

            # Keep prompt tokens fixed if provided
            if prompt_tokens is not None:
                x[0, :prompt_len] = prompt_tokens[0]

            if t % 100 == 0 or t == 0:
                generated_code = tokenizer.decode(x[0], skip_special_tokens=True)
                print(f"\nStep {t}:\n{generated_code}\n")

    # Final generated code
    generated_code = tokenizer.decode(x[0], skip_special_tokens=True)

    # Save the generated text to a file
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(generated_code)
        print(f"Generated text saved to {args.output_file}")

    return generated_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text Generation with Diffusion Models')
    parser.add_argument('--mode', choices=['train', 'generate'], required=True,
                        help='Whether to train the model or generate code')

    # Data and model parameters
    parser.add_argument('--data_path', type=str, help='Path to the training data')
    parser.add_argument('--model_path', type=str, help='Path to the trained model (for generation)')
    parser.add_argument('--save_path', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--output_file', type=str, help='File to save generated code')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers to use from the base model')
    parser.add_argument('--seq_length', type=int, default=512,
                        help='Maximum sequence length for training')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--num_down_steps', type=int, default=3,
                    help='Number of down/up steps in UNet')

    # Generation parameters
    parser.add_argument('--prompt', type=str, default=None,
                        help='Prompt for code generation')
    parser.add_argument('--gen_length', type=int, default=256,
                        help='Maximum length of generated code')
    parser.add_argument('--sampling_stride', type=int, default=10,
                        help='Stride for sampling steps (higher = faster, lower = better quality)')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'generate':
        generate(args)
