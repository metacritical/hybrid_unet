import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
import math
import json
import argparse
import os
from tqdm import tqdm
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions as in the original Transformer paper.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as described in the Transformer paper.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.out_proj(context)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Standard transformer block with pre-layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-LN architecture
        attn_input = self.attn_norm(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, mask)
        x = x + self.dropout(attn_output)
        
        ff_input = self.ff_norm(x)
        ff_output = self.ff(ff_input)
        x = x + self.dropout(ff_output)
        
        return x


class TimestepEmbedding(nn.Module):
    """
    Embeds timesteps for diffusion model conditioning.
    """
    def __init__(self, d_model):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model // 4)
        self.linear2 = nn.Linear(d_model // 4, d_model)
        
    def forward(self, t):
        # t: [batch_size] tensor of timesteps
        t = t.unsqueeze(-1).float()  # [batch_size, 1]
        t = F.silu(self.linear1(t))
        t = self.linear2(t)  # [batch_size, d_model]
        return t


class LLaDA(nn.Module):
    """
    LLaDA: Large Language Diffusion with mAsking
    A diffusion-based language model that uses masking.
    """
    def __init__(
        self,
        vocab_size=50257,  # GPT2 default vocab size
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_length=512,
        dropout=0.1,
        num_timesteps=1000
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.num_timesteps = num_timesteps
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        # Timestep embedding
        self.time_embedding = TimestepEmbedding(d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights if using the same vocab for input and output
        self.out_proj.weight = self.token_embedding.weight
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize weights with small values to improve training stability."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, timesteps):
        """
        Forward pass through the model.
        
        Args:
            x: [batch_size, seq_len] tensor of token indices or 
               [batch_size, seq_len, d_model] tensor of token embeddings
            timesteps: [batch_size] tensor of diffusion timesteps
            
        Returns:
            logits: [batch_size, seq_len, vocab_size] tensor of logits
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Embed tokens if x is indices
        if x.dtype == torch.long:
            x = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.position_encoding(x)
        
        # Add timestep embedding
        t_emb = self.time_embedding(timesteps)  # [batch_size, d_model]
        t_emb = t_emb.unsqueeze(1)  # [batch_size, 1, d_model]
        x = x + t_emb
        
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.norm(x)
        logits = self.out_proj(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TextDataset(Dataset):
    """
    Dataset for text diffusion model training.
    """
    def __init__(self, file_path, tokenizer, max_seq_length=512):
        self.max_seq_length = max_seq_length
        self.data = []
        self.tokenizer = tokenizer
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            for line in tqdm(f):
                try:
                    # Try to parse as JSON
                    item = json.loads(line)
                    if 'text' in item:
                        text = item['text']
                    elif 'content' in item:
                        text = item['content']
                    else:
                        # Use first string field
                        text = next((v for v in item.values() if isinstance(v, str)), "")
                except json.JSONDecodeError:
                    # Plain text
                    text = line.strip()
                    
                # Skip empty lines
                if not text.strip():
                    continue
                
                # Tokenize text
                if isinstance(tokenizer, tiktoken.Encoding):
                    tokens = tokenizer.encode(text)
                    if len(tokens) > max_seq_length:
                        tokens = tokens[:max_seq_length]
                    else:
                        tokens = tokens + [0] * (max_seq_length - len(tokens))
                    tokens = torch.tensor(tokens)
                else:
                    # Assuming HuggingFace tokenizer
                    tokens = tokenizer.encode(
                        text,
                        max_length=max_seq_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    ).squeeze(0)
                
                self.data.append(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def train(args):
    """
    Training loop for LLaDA model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up tokenizer
    if args.tokenizer_type == 'tiktoken':
        tokenizer = tiktoken.get_encoding(args.tiktoken_encoding)
        vocab_size = 100277 if args.tiktoken_encoding == 'cl100k_base' else 50257
    else:
        # Default to GPT2 tokenizer
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
    
    # Create model
    model = LLaDA(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_length=args.seq_length,
        dropout=args.dropout,
        num_timesteps=args.num_timesteps
    ).to(device)
    
    param_count = model.count_parameters()
    print(f"Model has {param_count:,} parameters")
    
    # Create dataset and dataloader
    dataset = TextDataset(args.data_path, tokenizer, args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    total_steps = len(dataloader) * args.epochs
    warmup_steps = args.warmup_ratio * total_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(device)
            
            # Generate random timesteps
            timesteps = torch.randint(
                0, args.num_timesteps, (batch.shape[0],), device=device
            )
            
            # Create masked tokens (using BERT-style format for clarity)
            masked_batch = batch.clone()
            
            # For each timestep t, mask tokens with probability t/num_timesteps
            for i, t in enumerate(timesteps):
                probs = torch.full(masked_batch[i].shape, float(t) / args.num_timesteps, device=device)
                mask = torch.bernoulli(probs).bool()
                
                # Get mask token ID
                if args.tokenizer_type == 'tiktoken':
                    mask_token_id = 0
                else:
                    mask_token_id = tokenizer.eos_token_id
                
                # Apply mask
                masked_batch[i] = torch.where(mask, torch.tensor(mask_token_id, device=device), masked_batch[i])
            
            # Get model predictions
            logits = model(masked_batch, timesteps)
            
            # Calculate cross-entropy loss only on masked tokens
            loss = 0
            for i, t in enumerate(timesteps):
                # Create mask of which tokens were masked
                mask_ratio = float(t) / args.num_timesteps
                mask = torch.bernoulli(torch.full(batch[i].shape, mask_ratio, device=device)).bool()
                
                if mask.sum() > 0:  # Only compute loss if there are masked tokens
                    # Get predictions and targets for masked tokens
                    masked_logits = logits[i][mask]
                    masked_targets = batch[i][mask]
                    
                    # Compute cross-entropy loss
                    loss += F.cross_entropy(
                        masked_logits,
                        masked_targets,
                        reduction='sum'
                    ) / mask.sum()
            
            # Average loss across batch
            loss = loss / batch.shape[0]
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
            
            # Save checkpoint periodically
            if (batch_idx + 1) % args.save_steps == 0:
                checkpoint_path = os.path.join(
                    args.save_path, f"checkpoint_epoch_{epoch+1}_step_{batch_idx+1}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'step': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save model after each epoch
        epoch_avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs} completed. Average loss: {epoch_avg_loss:.4f}")
        
        model_path = os.path.join(args.save_path, f"llada_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


def low_confidence_remasking(probs, num_to_remask):
    """
    Implementation of low-confidence remasking strategy from the LLaDA paper.
    Remasks tokens with lowest prediction confidence.
    
    Args:
        probs: token probabilities
        num_to_remask: number of tokens to remask
    
    Returns:
        indices of tokens to remask
    """
    confidences, _ = torch.max(probs, dim=-1)
    _, indices = torch.topk(confidences, k=num_to_remask, largest=False)
    return indices


def generate(args):
    """
    Text generation using LLaDA diffusion model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up tokenizer
    if args.tokenizer_type == 'tiktoken':
        tokenizer = tiktoken.get_encoding(args.tiktoken_encoding)
        vocab_size = 100277 if args.tiktoken_encoding == 'cl100k_base' else 50257
        
        # Define mask token ID
        mask_token_id = 0
        
        # Create a wrapper for tiktoken
        class TiktokenWrapper:
            def __init__(self, encoding):
                self.encoding = encoding
            
            def encode(self, text, return_tensors=None):
                tokens = self.encoding.encode(text)
                if return_tensors == 'pt':
                    return torch.tensor([tokens])
                return tokens
            
            def decode(self, tokens, skip_special_tokens=None):
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                return self.encoding.decode(tokens)
        
        tokenizer_wrapper = TiktokenWrapper(tokenizer)
        
    else:
        # Default to GPT2 tokenizer
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        mask_token_id = tokenizer.eos_token_id
        tokenizer_wrapper = tokenizer
    
    # Initialize model
    model = LLaDA(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_length=args.seq_length,
        dropout=0.0,  # No dropout during inference
        num_timesteps=args.num_timesteps
    ).to(device)
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Process prompt if provided
    if args.prompt:
        prompt_tokens = tokenizer_wrapper.encode(args.prompt, return_tensors='pt').to(device)
        prompt_len = prompt_tokens.size(1)
    else:
        prompt_tokens = None
        prompt_len = 0
    
    # Create fully masked sequence for generation
    x = torch.full((1, args.gen_length), mask_token_id, dtype=torch.long).to(device)
    
    # Copy prompt to the beginning if provided
    if prompt_tokens is not None:
        x[0, :prompt_len] = prompt_tokens[0]
    
    # Define time steps for sampling
    steps = torch.linspace(args.num_timesteps - 1, 0, args.sampling_steps + 1).long().to(device)
    
    print("Starting text generation with diffusion...")
    with torch.no_grad():
        for i in range(len(steps) - 1):
            current_t = steps[i]
            next_t = steps[i + 1]
            
            # Current denoising step
            timesteps = torch.full((1,), current_t, device=device)
            
            # Get logits from model
            logits = model(x, timesteps)
            probs = F.softmax(logits, dim=-1)
            
            # Identify which tokens are currently masked
            is_masked = (x == mask_token_id)
            
            if is_masked.any():
                # For masked tokens, get predictions
                pred_tokens = torch.argmax(probs, dim=-1)
                
                # Create new sequence with predictions for masked tokens
                new_x = x.clone()
                new_x[is_masked] = pred_tokens[is_masked]
                
                # For the next step, determine how many tokens should remain masked
                if next_t > 0:
                    current_mask_ratio = float(current_t) / args.num_timesteps
                    next_mask_ratio = float(next_t) / args.num_timesteps
                    
                    # Get number of tokens that should be remasked
                    tokens_to_unmask = is_masked.sum().item()
                    tokens_to_remask = int(tokens_to_unmask * next_mask_ratio / current_mask_ratio)
                    
                    if tokens_to_remask > 0:
                        # Get indices of tokens that were just unmasked
                        unmasked_indices = torch.where(is_masked)[1]
                        
                        # Either random remasking or low-confidence remasking
                        if args.remasking_strategy == 'random':
                            # Randomly select tokens to remask
                            perm = torch.randperm(len(unmasked_indices), device=device)
                            remask_indices = unmasked_indices[perm[:tokens_to_remask]]
                        else:  # 'low_confidence'
                            # Get probabilities for tokens that were masked
                            masked_probs = probs[0, is_masked.squeeze(0)]
                            
                            # Select tokens with lowest confidence
                            remask_local_indices = low_confidence_remasking(masked_probs, tokens_to_remask)
                            remask_indices = unmasked_indices[remask_local_indices]
                        
                        # Apply remasking
                        new_x[0, remask_indices] = mask_token_id
                
                x = new_x
            
            # Keep prompt fixed if provided
            if prompt_tokens is not None:
                x[0, :prompt_len] = prompt_tokens[0]
            
            # Log progress
            if (i + 1) % (len(steps) // 10) == 0 or i == len(steps) - 2:
                generated_text = tokenizer_wrapper.decode(x[0], skip_special_tokens=True)
                print(f"\nStep {i+1}/{len(steps)-1} (t={current_t}->{next_t}):")
                print(generated_text)
                print("-" * 40)
    
    # Final generated text
    generated_text = tokenizer_wrapper.decode(x[0], skip_special_tokens=True)
    
    # Save to file if specified
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(generated_text)
        print(f"Generated text saved to {args.output_file}")
    
    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaDA: Large Language Diffusion with Masking")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "generate"],
                      help="Whether to train the model or generate text")
    
    # Data and model paths
    parser.add_argument("--data_path", type=str, default=None,
                      help="Path to training data file")
    parser.add_argument("--model_path", type=str, default=None,
                      help="Path to load/save model checkpoints")
    parser.add_argument("--save_path", type=str, default="./checkpoints",
                      help="Directory to save model checkpoints")
    parser.add_argument("--output_file", type=str, default=None,
                      help="File to save generated text")
    
    # Model configuration
    parser.add_argument("--d_model", type=int, default=512,
                      help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=6,
                      help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                      help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048,
                      help="Feedforward dimension size")
    parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout probability")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                      help="Number of diffusion timesteps")
    
    # Tokenizer options
    parser.add_argument("--tokenizer_type", type=str, default="gpt2",
                      choices=["gpt2", "tiktoken"],
                      help="Type of tokenizer to use")
    parser.add_argument("--tiktoken_encoding", type=str, default="cl100k_base",
                      help="Tiktoken encoding name")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=10,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                      help="Portion of training to use for warmup")
    parser.add_argument("--seq_length", type=int, default=512,
                      help="Maximum sequence length")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Maximum gradient norm")
    parser.add_argument("--save_steps", type=int, default=1000,
                      help="Save checkpoint every X steps")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default=None,
                      help="Text prompt for generation")
    parser.add_argument("--gen_length", type=int, default=256,
                      help="Length of generated text")
    parser.add_argument("--sampling_steps", type=int, default=50,
                      help="Number of sampling steps")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence",
                      choices=["random", "low_confidence"],
                      help="Strategy for remasking during generation")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.data_path is None:
            parser.error("--data_path is required for training mode")
        train(args)
    elif args.mode == "generate":
        if args.model_path is None:
            parser.error("--model_path is required for generation mode")
        generate(args)
