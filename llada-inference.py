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
# Adjust this import to match your model file
from masked_diffusion_llm import LLaDA

# Initialize colorama
colorama.init()

def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA Text Generation with Real-time Visualization")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--d_model", type=int, default=512,
                        help="Hidden dimension size")
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048,
                        help="Feedforward dimension size")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    
    # Tokenizer options
    parser.add_argument("--tokenizer_type", type=str, default="gpt2",
                        choices=["gpt2", "tiktoken"],
                        help="Type of tokenizer to use")
    parser.add_argument("--tiktoken_encoding", type=str, default="cl100k_base",
                        help="Tiktoken encoding name")
    
    # Generation parameters
    parser.add_argument("--gen_length", type=int, default=256,
                        help="Length of generated text")
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="Number of sampling steps")
    parser.add_argument("--remasking_strategy", type=str, default="low_confidence",
                        choices=["random", "low_confidence"],
                        help="Strategy for remasking during generation")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (1.0 = no change, <1.0 = less random, >1.0 = more random)")
    parser.add_argument("--guidance_scale", type=float, default=3.0,
                        help="Scale for classifier-free guidance (0 = no guidance)")
    
    # Output options
    parser.add_argument("--output_file", type=str, default=None,
                        help="Save generated text to file")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay in seconds between visualization updates")
    parser.add_argument("--mask_symbol", type=str, default="▒",
                        help="Symbol to use for masked tokens in visualization")
    parser.add_argument("--highlight_new", action="store_true",
                        help="Highlight newly unmasked tokens")
    
    return parser.parse_args()


def setup_tokenizer(args):
    """Setup tokenizer and return tokenizer and mask token ID."""
    if args.tokenizer_type == 'tiktoken':
        encoding = tiktoken.get_encoding(args.tiktoken_encoding)
        vocab_size = 100277 if args.tiktoken_encoding == "cl100k_base" else 50257
        
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
                if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
                    tokens = tokens[0]  # Unwrap if nested list
                return self.encoding.decode(tokens)
        
        tokenizer = TiktokenWrapper(encoding)
        mask_token_id = 0  # Typically a reserved token in tiktoken
        
    else:  # Default to GPT2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        mask_token_id = tokenizer.eos_token_id  # Using EOS as the mask token for GPT2
    
    return tokenizer, vocab_size, mask_token_id


def low_confidence_remasking(probs, num_to_remask):
    """
    Implementation of low-confidence remasking strategy.
    Remasks tokens with lowest prediction confidence.
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
    print("LLaDA Text Generation Interactive Mode")
    print("=" * 60)
    print("Type a prompt and press Enter to generate text.")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 60 + "\n")


def temperature_sampling(logits, temperature=1.0):
    """
    Sample from logits with temperature control.
    Higher temperature increases randomness, lower temperature makes sampling more deterministic.
    """
    if temperature == 0:
        # Greedy sampling (argmax)
        return torch.argmax(logits, dim=-1)
    
    # Apply temperature
    logits = logits / max(temperature, 1e-5)  # Prevent division by zero
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample from the distribution
    return torch.multinomial(probs, 1).squeeze(-1)


def generate_text(model, tokenizer, mask_token_id, prompt=None, gen_length=256, 
                 sampling_steps=50, remasking_strategy="low_confidence", 
                 temperature=0.8, guidance_scale=3.0, num_timesteps=1000,
                 mask_symbol="▒", highlight_new=True, delay=0.1, device="cuda"):
    """
    Generate text with visualization of the unmasking process.
    
    Args:
        model: The LLaDA model
        tokenizer: Tokenizer for encoding/decoding
        mask_token_id: ID of the token used for masking
        prompt: Optional text prompt to start generation
        gen_length: Length of text to generate
        sampling_steps: Number of diffusion sampling steps
        remasking_strategy: Strategy for token remasking ("random" or "low_confidence")
        temperature: Temperature for sampling (1.0 = no change)
        guidance_scale: Scale for classifier-free guidance (0 = no guidance)
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
        if isinstance(tokenizer, GPT2Tokenizer):
            prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
        else:
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
    steps = torch.linspace(num_timesteps - 1, 0, sampling_steps + 1).long().to(device)
    
    # For visualization
    current_masked_positions = set(range(prompt_len, gen_length))
    previously_unmasked = set()
    
    # Initial visualization before generation
    if isinstance(tokenizer, GPT2Tokenizer):
        initial_text = tokenizer.decode(x[0], skip_special_tokens=True)
    else:
        initial_text = tokenizer.decode(x[0])
    
    # Pad the text with mask symbols to the gen_length
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
            timesteps = torch.full((1,), current_t, device=device)
            
            # Get logits from model
            logits = model(x, timesteps)
            
            # Optional: Apply classifier-free guidance
            if guidance_scale > 0:
                # Create an empty context input
                empty_context = torch.full_like(x, mask_token_id)
                if prompt_tokens is not None:
                    empty_context[0, :prompt_len] = prompt_tokens[0]
                
                # Get unconditional logits
                uncond_logits = model(empty_context, timesteps)
                
                # Apply guidance
                logits = uncond_logits + guidance_scale * (logits - uncond_logits)
            
            # Identify which tokens are currently masked
            is_masked = (x == mask_token_id).squeeze(0)
            
            if is_masked.any():
                # Get predicted tokens with temperature
                if temperature <= 0:
                    pred_tokens = torch.argmax(logits, dim=-1)
                else:
                    # Apply temperature sampling
                    pred_tokens = temperature_sampling(logits, temperature)
                
                # Create new sequence with predictions for masked tokens
                new_x = x.clone()
                new_x[0, is_masked] = pred_tokens[0, is_masked]
                
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
                        if remasking_strategy == 'random':
                            # Randomly select tokens to remask
                            perm = torch.randperm(len(unmasked_indices), device=device)
                            remask_indices = unmasked_indices[perm[:tokens_to_remask]]
                        else:  # 'low_confidence'
                            # Get probabilities for unmasked tokens
                            masked_probs = F.softmax(logits[0, is_masked], dim=-1)
                            
                            # Select tokens with lowest confidence
                            if len(masked_probs) > tokens_to_remask:
                                remask_local_indices = low_confidence_remasking(masked_probs, tokens_to_remask)
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
                if isinstance(tokenizer, GPT2Tokenizer):
                    current_text = tokenizer.decode(x[0], skip_special_tokens=True)
                else:
                    current_text = tokenizer.decode(x[0])
                
                # Pad the text if needed
                if len(current_text) < gen_length:
                    current_text = current_text + " " * (gen_length - len(current_text))
                
                # Visualize
                visualize_generation(current_text, current_masked_positions, newly_highlighted, mask_symbol)
                time.sleep(delay)
    
    # Final display with no masks
    if isinstance(tokenizer, GPT2Tokenizer):
        final_text = tokenizer.decode(x[0], skip_special_tokens=True)
    else:
        final_text = tokenizer.decode(x[0])
    
    # Show final result
    print("\n\nFinal text:")
    print("-" * 60)
    print(final_text)
    print("-" * 60)
    
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
            generate_text(
                model=model,
                tokenizer=tokenizer,
                mask_token_id=mask_token_id,
                prompt=user_prompt,
                gen_length=args.gen_length,
                sampling_steps=args.sampling_steps,
                remasking_strategy=args.remasking_strategy,
                temperature=args.temperature,
                guidance_scale=args.guidance_scale,
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
                    f.write(f"Generated: {final_text}\n\n")
                print(f"Output appended to {args.output_file}")
            
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up tokenizer
    tokenizer, vocab_size, mask_token_id = setup_tokenizer(args)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = LLaDA(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_length=args.gen_length,
        dropout=0.0,  # No dropout during inference
        num_timesteps=args.num_timesteps
    ).to(device)
    
    # Load checkpoint with proper safety settings
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    
    # Start interactive session
    interactive_session(model, tokenizer, mask_token_id, args, device)


if __name__ == "__main__":
    main()
