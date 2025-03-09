import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from diffusers import DDPMScheduler
import json
import argparse
from tqdm import tqdm
import os

class CodeUNet(nn.Module):
    """
    UNet architecture specialized for code understanding and generation.
    Based on the insights from the LLaDA paper's bidirectional approach.
    """
    def __init__(self, embed_dim, seq_length):
        super().__init__()
        self.embed_dim = embed_dim
        
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
        
        # Middle block with dilated convolutions for broader context
        self.middle = nn.Sequential(
            nn.Conv2d(embed_dim*4, embed_dim*4, kernel_size=(1,3), padding=(0,2), dilation=(1,2)),
            nn.SiLU(),
            nn.Conv2d(embed_dim*4, embed_dim*4, kernel_size=(1,3), padding=(0,1))
        )
        
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
        
        # Downsample path
        x1 = self.down1(x)      # [B, C*2, 1, S/2]
        skip2 = x1
        
        x2 = self.down2(x1)     # [B, C*4, 1, S/4]
        
        # Middle
        x2 = self.middle(x2)    # [B, C*4, 1, S/4]
        
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
        
        # Custom UNet for diffusion process
        self.unet = CodeUNet(self.embed_dim, seq_length)
        
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
    
    model.print_params()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_timesteps)
    
    dataset = TextDataset(args.data_path, tokenizer, args.seq_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Training loop with diffusion process
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
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
        print(f'\nEpoch {epoch+1}/{args.epochs} completed. Average Loss: {avg_loss:.4f}')
        
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
