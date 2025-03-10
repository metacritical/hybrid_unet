#!/usr/bin/env python
import torch
import torch.nn.functional as F
import argparse
import os
import time
from transformers import GPT2Tokenizer
import tiktoken
import colorama
from colorama import Fore, Back, Style
import re
import sys

# Import the model architecture from your implementation
# This assumes your model is in code-diffusion-mh_attention-model.py
# Modify this to match your actual module name and class
from code_diffusion_mh_attention_model import TextDiffusionModel, DiffusionUNet, TransformerBlock, MultiHeadAttention

# Initialize colorama
colorama.init()

def parse_args():
    parser = argparse.ArgumentParser(description="MultiHead Attention Text Diffusion Model Inference")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--num_layers", type=int, default=6, 
                        help="Number of transformer layers")
    parser.add_argument("--seq_length", type=int, default=512, 
                        help="Maximum sequence length for training")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    
    # Tokenizer options
    parser.add_argument("--tokenizer_type", type=str, default="gpt2",
                        choices=["gpt2", "tiktoken"],
                        help="Type of tokenizer to use")
    parser.add_argument("--tiktoken_encoding", type=str, default="cl100k_base",
                        help="Tiktoken encoding to use (cl100k_base for GPT-4, p50k_base for GPT-3)")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to start generation with")
    parser.add_argument("--gen_length", type=int, default=256,
                        help="Maximum length of generated text")
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="Number of sampling steps")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Only sample from the top k most likely tokens")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Only sample from tokens with cumulative probability < top_p")
    
    # Visualization options
    parser.add_argument("--output_file", type=str, default=None,
                        help="File to save generated text")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay between visualization updates (seconds)")
    parser.add_argument("--mask_symbol", type=str, default="▒",
                        help="Symbol to use for masked tokens")
    parser.add_argument("--highlight_new", action="store_true",
                        help="Highlight newly unmasked tokens")
    
    return parser.parse_args()


def setup_tokenizer(args):
    """Setup tokenizer and return tokenizer and mask token ID."""
    if args.tokenizer_type == 'tiktoken':
        encoding = tiktoken.get_encoding(args.tiktoken_encoding)
        vocab_size = 100277 if args.tiktoken_encoding == "cl100k_base" else 50257
        
        # Create a simple wrapper to match the HF tokenizer interface
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
        
        tokenizer = TiktokenWrapper(encoding)
        mask_token_id = 0  # Typically a reserved token in tiktoken
    else:  # Default to GPT2
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        # GPT2 doesn't have a mask token, so we'll use a convention
        mask_token_id = tokenizer.eos_token_id  # Using EOS as the mask token
        
    return tokenizer, vocab_size, mask_token_id


def low_confidence_remasking(probs, num_to_remask):
    """
    Implementation of low-confidence remasking strategy from the paper.
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


def visualize_generation(current_text, masked_positions, newly_unmasked=None, mask_symbol="▒"):
    """Visualize the current state of text generation with masking."""
    # Clear the line
    sys.stdout.write("\r" + " " * 100 + "\r")
    
    # Create a list for the result
    result = []
    
    for i, char in enumerate(current_text):
        if i in masked_positions:
            result.append(Fore.CYAN + mask_symbol + Style.RESET_ALL)
        elif newly_unmasked and i in newly_unmasked:
            result.append(Fore.GREEN + char + Style.RESET_ALL)
        else:
            result.append(char)
    
    # Join and print the result
    sys.stdout.write("".join(result))
    sys.stdout.flush()


def create_interactive_prompt():
    """Create an interactive prompt UI for text generation."""
    print("\n" + "=" * 60)
    print("MultiHead Attention Text Diffusion - Interactive Mode")
    print("=" * 60)
    print("Type a prompt and press Enter to generate text.")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 60 + "\n")


def apply_top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("inf")):
    """
    Apply top-k and top-p (nucleus) filtering to logits.
    
    Args:
        logits: Logits to modify (B, vocab_size)
        top_k: Keep only the top k tokens with highest probability (top-k filtering)
        top_p: Keep the top tokens with cumulative probability >= top_p (nucleus filtering)
        filter_value: Value to assign to filtered tokens
    
    Returns:
        Modified logits
    """
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter back the indices to the original logits tensor
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def temperature_sampling(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    Sample from logits with temperature, top-k, and top-p controls.
    
    Args:
        logits: Logits to sample from (batch_size, sequence_length, vocab_size)
        temperature: Temperature for sampling
        top_k: Number of highest probability tokens to keep
        top_p: Cumulative probability threshold for nucleus sampling
    
    Returns:
        Sampled token indices
    """
    if temperature == 0:
        # Greedy sampling (argmax)
        return torch.argmax(logits, dim=-1)
    
    # Apply temperature
    logits = logits / max(temperature, 1e-5)  # Prevent division by zero
    
    # Apply top-k and top-p filtering
    filtered_logits = apply_top_k_top_p_filtering(
        logits.clone(), top_k=top_k, top_p=top_p
    )
    
    # Convert to probabilities
    probs = F.softmax(filtered_logits, dim=-1)
    
    # Sample from the distribution
    return torch.multinomial(probs, 1).squeeze(-1)

def get_vocab_size_from_checkpoint(checkpoint_path, device):
    """Extract vocabulary size from a model checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'token_emb.weight' in state_dict:
        return state_dict['token_emb.weight'].size(0)
    else:
        # Default to GPT2 vocab size if not found
        return 50257


def generate_text(model, tokenizer, mask_token_id, prompt=None, gen_length=256, 
                  sampling_steps=50, temperature=0.8, top_k=50, top_p=0.95,
                  num_timesteps=1000, mask_symbol="▒", highlight_new=True, 
                  delay=0.1, device="cuda"):
    """
    Generate text with visualization of the unmasking process.
    
    Args:
        model: The diffusion model
        tokenizer: Tokenizer for encoding/decoding
        mask_token_id: ID of the token used for masking
        prompt: Optional text prompt to start generation
        gen_length: Length of text to generate
        sampling_steps: Number of diffusion sampling steps
        temperature: Temperature for sampling (1.0 = no change)
        top_k: Keep only top k tokens with highest probability (0 to disable)
        top_p: Keep top tokens with cumulative probability < top_p (1.0 to disable)
        num_timesteps: Total number of diffusion timesteps
        mask_symbol: Symbol to display for masked tokens
        highlight_new: Whether to highlight newly unmasked tokens
        delay: Delay between visualization updates
        device: Device to run generation on
        
    Returns:
        Generated text
    """
    model.eval()
    
    # Process prompt if provided
    if prompt:
        prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_len = prompt_tokens.size(1)
    else:
        prompt_tokens = None
        prompt_len = 0
    
    # Create fully masked sequence for generation
    x = torch.full((1, gen_length), mask_token_id, dtype=torch.long).to(device)
    
    # Copy prompt to the beginning if provided
    if prompt_tokens is not None:
        x[0, :prompt_len] = prompt_tokens[0]
    
    # Define time steps for sampling
    # Convert from [0, 1, 2, ..., num_timesteps-1] to [1.0, 0.9, 0.8, ..., 0.0]
    steps = torch.linspace(num_timesteps - 1, 0, sampling_steps + 1).long().to(device)
    
    # For visualization
    current_masked_positions = set(range(prompt_len, gen_length))
    previously_unmasked = set()
    
    # Initial visualization before generation
    initial_text = tokenizer.decode(x[0], skip_special_tokens=True)
    
    # Pad the text to the desired length
    padded_text = initial_text + " " * (gen_length - len(initial_text))
    
    # Visualize initial state
    visualize_generation(padded_text, current_masked_positions, None, mask_symbol)
    time.sleep(delay)
    
    print("\nStarting text generation with diffusion...")
    
    with torch.no_grad():
        for i in range(len(steps) - 1):
            current_t = steps[i]
            next_t = steps[i + 1]
            
            # Current denoising step
            timesteps = torch.tensor([current_t], device=device)
            
            # Get model predictions
            logits = model(x, timesteps)
            
            # Identify which tokens are currently masked
            is_masked = (x == mask_token_id).squeeze(0)
            
            if is_masked.any():
                # For masked tokens, sample new tokens with temperature, top-k, and top-p
                masked_logits = logits[:, is_masked, :]
                sampled_tokens = temperature_sampling(
                    masked_logits.view(-1, logits.size(-1)), 
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # Create new sequence with sampled tokens
                new_x = x.clone()
                new_x[0, is_masked] = sampled_tokens
                
                # For visualization: get indices of newly unmasked tokens
                newly_unmasked_indices = set()
                for idx in range(gen_length):
                    if idx in current_masked_positions and new_x[0, idx] != mask_token_id:
                        newly_unmasked_indices.add(idx)
                
                # For the next step, determine how many tokens should remain masked
                if next_t > 0:
                    current_mask_ratio = float(current_t) / num_timesteps
                    next_mask_ratio = float(next_t) / num_timesteps
                    
                    # Get number of tokens that should be remasked
                    unmasked_indices = torch.where(is_masked)[0]
                    tokens_to_unmask = len(unmasked_indices)
                    tokens_to_remask = int(tokens_to_unmask * next_mask_ratio / current_mask_ratio)
                    
                    if tokens_to_remask > 0 and len(unmasked_indices) > 0:
                        # Get probabilities for tokens that were just unmasked
                        probs = F.softmax(masked_logits.squeeze(0), dim=-1)
                        
                        # Get the predicted token probabilities
                        token_probs = torch.gather(
                            probs, 1, 
                            sampled_tokens.unsqueeze(-1)
                        ).squeeze(-1)
                        
                        # Get low confidence tokens to remask
                        remask_local_indices = low_confidence_remasking(
                            token_probs.unsqueeze(0), tokens_to_remask
                        )
                        
                        # Convert local indices to global indices
                        if len(remask_local_indices) > 0:
                            remask_global_indices = unmasked_indices[remask_local_indices]
                            
                            # Apply remasking
                            new_x[0, remask_global_indices] = mask_token_id
                
                x = new_x
                
                # Update masked positions for visualization
                current_masked_positions = set(idx.item() for idx in torch.where(x[0] == mask_token_id)[0])
                
                # Update set of previously unmasked tokens
                if highlight_new:
                    newly_highlighted = newly_unmasked_indices - previously_unmasked
                    previously_unmasked.update(newly_unmasked_indices)
                else:
                    newly_highlighted = None
                
                # Keep prompt fixed if provided
                if prompt_tokens is not None:
                    x[0, :prompt_len] = prompt_tokens[0]
                
                # Visualize the current state
                current_text = tokenizer.decode(x[0], skip_special_tokens=True)
                
                # Pad the text if needed
                if len(current_text) < gen_length:
                    current_text = current_text + " " * (gen_length - len(current_text))
                
                # Visualize
                visualize_generation(current_text, current_masked_positions, newly_highlighted, mask_symbol)
                time.sleep(delay)
    
    # Final decoded text
    final_text = tokenizer.decode(x[0], skip_special_tokens=True)
    
    # Show final result
    print("\n\nFinal text:")
    print("-" * 80)
    print(final_text)
    print("-" * 80)
    
    return final_text


def interactive_session(model, tokenizer, mask_token_id, args, device):
    """Run an interactive session for text generation."""
    create_interactive_prompt()
    
    while True:
        try:
            # Get user input
            user_prompt = input("\nEnter prompt (or 'exit' to quit): ")
            
            if user_prompt.lower() in ['exit', 'quit']:
                print("\nExiting interactive session.")
                break
            
            # Generate text
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                mask_token_id=mask_token_id,
                prompt=user_prompt,
                gen_length=args.gen_length,
                sampling_steps=args.sampling_steps,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_timesteps=args.num_timesteps,
                mask_symbol=args.mask_symbol,
                highlight_new=args.highlight_new,
                delay=args.delay,
                device=device
            )
            
            # Save to file if requested
            if args.output_file:
                with open(args.output_file, 'a') as f:
                    f.write(f"Prompt: {user_prompt}\n")
                    f.write(f"Generated: {generated_text}\n\n")
                print(f"Output appended to {args.output_file}")
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up tokenizer
    checkpoint_vocab_size = get_vocab_size_from_checkpoint(args.model_path, device)

    tokenizer, _, mask_token_id = setup_tokenizer(args)
    
    # Initialize model
    print(f"Loading model from {args.model_path}")
    model = TextDiffusionModel(
        vocab_size=checkpoint_vocab_size,
        num_layers=args.num_layers,
        seq_length=args.seq_length,
        num_timesteps=args.num_timesteps
    ).to(device)
    
    # Load model with safety settings
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Print model summary
    print(f"Detected vocabulary size from checkpoint: {checkpoint_vocab_size}")

    
    # Start interactive session
    interactive_session(model, tokenizer, mask_token_id, args, device)


if __name__ == "__main__":
    main()
