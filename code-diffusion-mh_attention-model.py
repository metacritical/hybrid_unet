import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from diffusers import DDPMScheduler
import json
import argparse
from tqdm import tqdm
import os

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module similar to Transformer architecture.
    This is essential for diffusion models to capture complex token relationships.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Scaling factor for dot product attention
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention weights computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        # Final projection
        return self.out_proj(attn_output)


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network.
    Core building block for diffusion-based language modeling.
    """
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # First sublayer: multi-head attention with residual connection
        attn_output = self.attn(self.norm1(x))
        x = x + attn_output

        # Second sublayer: feed-forward network with residual connection
        ff_output = self.ff(self.norm2(x))
        return x + ff_output


class DiffusionUNet(nn.Module):
    """
    UNet architecture with transformer blocks for diffusion-based text generation.
    Incorporates attention mechanisms as described in the LLaDA paper.
    """
    def __init__(self, embed_dim, seq_length, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim

        # Attention blocks for each level
        self.attn1 = TransformerBlock(embed_dim, num_heads, embed_dim * 4)
        self.attn2 = TransformerBlock(embed_dim * 2, num_heads, embed_dim * 8)
        self.attn3 = TransformerBlock(embed_dim * 4, num_heads, embed_dim * 16)

        # Downsample blocks - extracting features at different scales
        self.down1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim*2, kernel_size=(1,3), padding=(0,1)),
            nn.SiLU(),  # Similar to Mish but more stable
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim*4, kernel_size=(1,3), padding=(0,1)),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        )

        # Middle transformer block for global context modeling
        self.middle_transformer = TransformerBlock(embed_dim * 4, num_heads, embed_dim * 16)

        # Upsample blocks - reconstructing with skip connections
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim*8, embed_dim*2,
                              kernel_size=(1,3), stride=(1,2),
                              padding=(0,1), output_padding=(0,1)),
            nn.SiLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim*4, embed_dim,
                              kernel_size=(1,3), stride=(1,2),
                              padding=(0,1), output_padding=(0,1)),
            nn.SiLU()
        )

        self.final = nn.Conv2d(embed_dim*2, embed_dim, kernel_size=(1,1))

        # Time embedding to condition on diffusion timesteps
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim//4),
            nn.SiLU(),
            nn.Linear(embed_dim//4, embed_dim)
        )

    def forward(self, x, timesteps):
        # x shape: [B, C, 1, S] - batch, channels, height (1), sequence length
        batch_size = x.size(0)

        # Embed timesteps and add to input as a conditioning signal
        t_emb = self.time_embed(timesteps.view(-1, 1).float()).view(batch_size, self.embed_dim, 1, 1)
        x = x + t_emb

        # Skip connections
        skip1 = x

        # Apply attention at the first level (reshape to [B, S, C])
        x_attn = x.squeeze(2).permute(0, 2, 1)  # [B, S, C]
        x_attn = self.attn1(x_attn)
        x = x_attn.permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, S]

        # Downsample path
        x1 = self.down1(x)      # [B, C*2, 1, S/2]
        skip2 = x1

        # Apply attention at the second level
        x1_attn = x1.squeeze(2).permute(0, 2, 1)  # [B, S/2, C*2]
        x1_attn = self.attn2(x1_attn)
        x1 = x1_attn.permute(0, 2, 1).unsqueeze(2)  # [B, C*2, 1, S/2]

        x2 = self.down2(x1)     # [B, C*4, 1, S/4]

        # Apply attention at the bottleneck
        x2_attn = x2.squeeze(2).permute(0, 2, 1)  # [B, S/4, C*4]
        x2_attn = self.attn3(x2_attn)
        x2 = x2_attn.permute(0, 2, 1).unsqueeze(2)  # [B, C*4, 1, S/4]

        # Middle - additional global context with transformer
        x2_middle = x2.squeeze(2).permute(0, 2, 1)  # [B, S/4, C*4]
        x2_middle = self.middle_transformer(x2_middle)
        x2 = x2_middle.permute(0, 2, 1).unsqueeze(2)  # [B, C*4, 1, S/4]

        # Upsample path with skip connections
        x = torch.cat([x2, x2], dim=1)  # Concatenate to match channel dim
        x = self.up1(x)         # [B, C*2, 1, S/2]

        x = torch.cat([x, skip2], dim=1)  # Skip connection
        x = self.up2(x)         # [B, C, 1, S]

        x = torch.cat([x, skip1], dim=1)  # Skip connection
        x = self.final(x)       # [B, C, 1, S]

        return x


class TextDiffusionModel(nn.Module):
    """
    A diffusion-based text generation model that combines a pre-trained
    language model (BERT) with a UNet for bidirectional understanding.
    """
    def __init__(
        self,
        vocab_size=30522,  # BERT vocabulary size
        seq_length=512,
        num_layers=6,
        num_timesteps=1000
    ):
        super().__init__()

        # Use BERT as the base model for text understanding
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.embed_dim = self.config.hidden_size
        self.seq_length = seq_length

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, self.embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_length, self.embed_dim))

        # BERT layers for context understanding
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_layers = self.text_model.encoder.layer[:num_layers]

        # Custom UNet with attention mechanisms for diffusion process
        self.unet = DiffusionUNet(self.embed_dim, seq_length, num_heads=8)

        # Final projection to vocabulary
        self.to_logits = nn.Linear(self.embed_dim, vocab_size)

        # Embedding for timesteps
        self.timestep_emb = nn.Embedding(num_timesteps, self.embed_dim)

        # Layer norm for stabilization
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x, timesteps):
        """Main forward pass handling both token indices and embeddings"""
        batch_size = x.size(0)

        if x.dtype == torch.long:
            # For token indices, first embed them
            x = self.token_emb(x) + self.pos_emb[:, :x.size(1), :]
        else:
            # For already embedded tokens, just add position embeddings
            x = x + self.pos_emb[:, :x.size(1), :]

        # Add timestep embedding for diffusion conditioning
        t_emb = self.timestep_emb(timesteps).unsqueeze(1)
        x += t_emb

        # BERT processing for text understanding
        for layer in self.text_layers:
            x = layer(x)[0]

        # Reshape for UNet processing
        x = x.permute(0, 2, 1).contiguous().unsqueeze(2)  # [B, C, 1, S]

        # UNet processing with diffusion timesteps
        x = self.unet(x, timesteps)

        # Reshape back to sequence format
        x = x.squeeze(2).permute(0, 2, 1).contiguous()

        # Normalize and project to vocabulary
        x = self.norm(x)
        return self.to_logits(x)

    def forward_tokens(self, tokens, timesteps):
        """Process token indices through the model"""
        batch_size = tokens.size(0)

        # Text embedding pipeline
        x = self.token_emb(tokens) + self.pos_emb[:, :tokens.size(1), :]
        t_emb = self.timestep_emb(timesteps).unsqueeze(1)
        x += t_emb

        # BERT processing
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
    """
    Dataset for text training with efficient masking for diffusion.
    Handles both plain text and structured documents.
    """
    def __init__(self, file_path, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.data = []

        print(f"Loading text data from {file_path}...")
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc="Processing dataset"):
                try:
                    # Try to parse as JSON if the data is structured
                    parsed = json.loads(line)
                    # Look for common text fields
                    if 'text' in parsed:
                        text = parsed['text']
                    elif 'content' in parsed:
                        text = parsed['content']
                    elif 'body' in parsed:
                        text = parsed['body']
                    else:
                        # Use the first string value found
                        text = next((v for v in parsed.values() if isinstance(v, str)), "")
                except json.JSONDecodeError:
                    # If not JSON, treat the line as plain text
                    text = line

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
    """Training function with diffusion-based training procedure"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Initializing tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = TextDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        num_timesteps=args.num_timesteps
    ).to(device)

    # Load pretrained model if specified
    start_epoch = 0
    if args.pretrained_model:
        if os.path.exists(args.pretrained_model):
            print(f"Loading pretrained model from {args.pretrained_model}")
            try:
                # Load model state dict
                state_dict = torch.load(args.pretrained_model, map_location=device)
                model.load_state_dict(state_dict)

                # Extract epoch number from filename if possible
                filename = os.path.basename(args.pretrained_model)
                if "epoch_" in filename:
                    try:
                        epoch_str = filename.split("epoch_")[1].split(".")[0]
                        start_epoch = int(epoch_str)
                        print(f"Continuing training from epoch {start_epoch}")
                    except (ValueError, IndexError):
                        print("Could not determine epoch from filename. Starting from epoch 0.")

                print("Pretrained model loaded successfully.")
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                print("Initializing from scratch instead.")
        else:
            print(f"Pretrained model file {args.pretrained_model} not found. Initializing from scratch.")

    model.print_params()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)

    dataset = TextDataset(args.data_path, tokenizer, args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)

    # Training loop with diffusion process
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{start_epoch + args.epochs}')

        for batch in progress_bar:
            batch = batch.to(device)

            # Generate random timesteps for diffusion
            timesteps = torch.randint(0, args.num_timesteps, (batch.size(0),), device=device)

            # Get clean token embeddings
            with torch.no_grad():
                clean_embeddings = model.token_emb(batch)

            # Add noise according to the scheduler
            noise = torch.randn_like(clean_embeddings)
            noisy_embeddings = scheduler.add_noise(clean_embeddings, noise, timesteps)

            # Pass the noisy embeddings to predict the original tokens
            logits = model.forward_embeddings(noisy_embeddings, timesteps)

            # Cross entropy loss only on masked (noised) tokens
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch.reshape(-1),
                ignore_index=tokenizer.pad_token_id
            )

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f'\nEpoch {epoch+1}/{start_epoch + args.epochs} completed. Average Loss: {avg_loss:.4f}')

        # Save model checkpoint
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = TextDiffusionModel(
        vocab_size=tokenizer.vocab_size,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        num_timesteps=args.num_timesteps
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
                temperature = 0.5  # Adjust as needed
                probs = probs ** (1/temperature)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                pred_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.size(0), -1)


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
    parser.add_argument('--pretrained_model', type=str, help='Path to a pretrained model to continue training from')

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
