#!/usr/bin/env python
"""
nano - Interactive chat interface
A minimal GPT implementation for educational purposes
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tokenizers import Tokenizer

from src.config import ModelConfig
from src.model import GPT


# Neon colors for terminal
class Colors:
    HEADER = '\033[95m'      # Purple
    BLUE = '\033[94m'        # Blue  
    CYAN = '\033[96m'        # Cyan
    GREEN = '\033[92m'       # Green
    YELLOW = '\033[93m'      # Yellow
    RED = '\033[91m'         # Red
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']


def load_checkpoint(filepath, device):
    """Load checkpoint using project model loading logic"""
    filepath = Path(filepath)
    
    checkpoint = torch.load(
        filepath, 
        map_location=device,
        weights_only=False
    )
    
    # Get config from checkpoint or use default
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = ModelConfig()
    
    # Load tokenizer
    tokenizer_path = Path("data/tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError("Tokenizer not found! Run tokenizer training first.")
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # Tokenizer now has ByteLevel decoder built-in (configured in train_tokenizer.py)
    config.vocab_size = tokenizer.get_vocab_size()
    
    # Create and load model
    model = GPT(config)
    
    # Handle torch.compile() prefix in checkpoint
    state_dict = checkpoint["model"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    step = checkpoint.get("step", "unknown")
    loss = checkpoint.get("best_val_loss", "unknown")
    
    return model, tokenizer, config, step, loss


def generate_response(model, tokenizer, prompt, device, temperature=0.8, top_k=40, top_p=0.9, max_tokens=200, show_thinking=True):
    """Generate response using model.generate() with typewriter effect"""
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    prompt_length = len(encoded.ids)
    
    # Generate using model's built-in generate method
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if top_p > 0.0 else None,
            )
    
    # Decode full output
    generated_ids = output_ids[0].tolist()
    full_text = tokenizer.decode(generated_ids)
    
    # Extract only the generated part (after the prompt)
    # We need to decode the prompt separately to get exact string match
    prompt_text = tokenizer.decode(encoded.ids)
    response_text = full_text[len(prompt_text):]
    
    # Stream the response with typewriter effect
    if show_thinking:
        for char in response_text:
            print(char, end='', flush=True)
            time.sleep(0.01)
    
    return response_text


def get_available_models(checkpoints_dir="checkpoints"):
    """Get list of available checkpoint files"""
    models = []
    checkpoints_path = Path(checkpoints_dir)
    if checkpoints_path.exists():
        for ckpt in sorted(checkpoints_path.glob("*.pt")):
            models.append(ckpt.stem)
    return models


def print_banner():
    """Print gothic banner"""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   {Colors.BOLD}{Colors.GREEN} ██████╗  ██████╗ ████████╗██╗  ██╗██╗ ██████╗{Colors.CYAN}                             ║
║   {Colors.BOLD}{Colors.GREEN}██╔════╝ ██╔═══██╗╚══██╔══╝██║  ██║██║██╔════╝{Colors.CYAN}                             ║
║   {Colors.BOLD}{Colors.GREEN}██║  ███╗██║   ██║   ██║   ███████║██║██║     {Colors.CYAN}                             ║
║   {Colors.BOLD}{Colors.GREEN}██║   ██║██║   ██║   ██║   ██╔══██║██║██║     {Colors.CYAN}                             ║
║   {Colors.BOLD}{Colors.GREEN}╚██████╔╝╚██████╔╝   ██║   ██║  ██║██║╚██████╗{Colors.CYAN}                             ║
║   {Colors.BOLD}{Colors.GREEN} ╚═════╝  ╚═════╝    ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝{Colors.CYAN}                             ║
║                                                                              ║
║   {Colors.YELLOW}Interactive Language Model Interface{Colors.CYAN}                                        ║
║   {Colors.YELLOW}Pure PyTorch • Minimal • Fast{Colors.CYAN}                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}

{Colors.CYAN}Commands:{Colors.END}
  {Colors.GREEN}/model <name>{Colors.END}    - Switch to checkpoint
  {Colors.GREEN}/models{Colors.END}          - List available models
  {Colors.GREEN}/temp <0.1-2.0>{Colors.END}  - Adjust creativity (default: 0.8)
  {Colors.GREEN}/topk <1-100>{Colors.END}    - Set top-k sampling (default: 40)
  {Colors.GREEN}/topp <0.0-1.0>{Colors.END}  - Set top-p sampling (default: 0.9)
  {Colors.GREEN}/max <10-500>{Colors.END}    - Set max tokens (default: 200)
  {Colors.GREEN}/reset{Colors.END}           - Clear conversation history
  {Colors.GREEN}/params{Colors.END}          - Show model statistics
  {Colors.GREEN}/help{Colors.END}            - Show all commands
  {Colors.GREEN}/quit{Colors.END}            - Exit gracefully

{Colors.YELLOW}Start typing to chat...{Colors.END}
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="nano - Interactive Language Model")
    parser.add_argument("--checkpoint", help="Path to checkpoint file (optional)")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (0.1-2.0)")
    parser.add_argument("--topk", type=int, default=40, help="Top-k sampling (1-100)")
    parser.add_argument("--topp", type=float, default=0.9, help="Top-p sampling (0.0-1.0)")
    parser.add_argument("--max", type=int, default=200, help="Max tokens (10-500)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize variables
    temperature = args.temp
    top_k = args.topk
    top_p = args.topp
    max_tokens = args.max
    conversation_history = []
    current_checkpoint = [None]  # Use list for mutability
    model = None
    tokenizer = None
    config = None
    
    # Load model if checkpoint provided
    if args.checkpoint:
        print(f"\n{Colors.CYAN}Loading checkpoint: {args.checkpoint}...{Colors.END}")
        try:
            model, tokenizer, config, step, loss = load_checkpoint(args.checkpoint, device)
            current_checkpoint[0] = args.checkpoint
            print(f"{Colors.GREEN}✓ Model loaded!{Colors.END}")
            print(f"{Colors.CYAN}  Step: {step} | Loss: {loss:.4f}{Colors.END}" if isinstance(loss, float) else f"{Colors.CYAN}  Step: {step}{Colors.END}")
            print(f"{Colors.CYAN}  Device: {device} | Vocab: {config.vocab_size} tokens{Colors.END}\n")
        except Exception as e:
            print(f"{Colors.RED}✗ Failed to load model: {e}{Colors.END}")
            sys.exit(1)
    else:
        print(f"\n{Colors.YELLOW}No checkpoint provided.{Colors.END}")
        print(f"{Colors.CYAN}Use {Colors.GREEN}/models{Colors.CYAN} to list available models{Colors.END}")
        print(f"{Colors.CYAN}Use {Colors.GREEN}/model <name>{Colors.CYAN} to load a model{Colors.END}\n")
    
    print_banner()
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n{Colors.GREEN}You{Colors.END} {Colors.CYAN}➜{Colors.END} ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                if user_input == '/quit' or user_input == '/exit':
                    print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.END}\n")
                    break
                
                elif user_input == '/reset' or user_input == '/clear':
                    conversation_history = []
                    print(f"{Colors.CYAN}✓ Conversation history cleared{Colors.END}")
                    continue
                
                elif user_input == '/params':
                    if model is None:
                        print(f"{Colors.YELLOW}No model loaded. Use /model <name> to load one.{Colors.END}")
                        continue
                    print(f"\n{Colors.CYAN}Model Parameters:{Colors.END}")
                    print(f"  Architecture: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
                    print(f"  Block size: {config.block_size}")
                    print(f"  Vocabulary: {config.vocab_size} tokens")
                    print(f"  Parameters: {model.count_parameters():,}")
                    print(f"  Checkpoint: {Path(current_checkpoint[0]).name if current_checkpoint[0] else 'None'}")
                    print(f"  Device: {device}")
                    print(f"\n{Colors.CYAN}Generation Settings:{Colors.END}")
                    print(f"  Temperature: {temperature}")
                    print(f"  Top-k: {top_k}")
                    print(f"  Top-p: {top_p}")
                    print(f"  Max tokens: {max_tokens}")
                    print(f"  Total history: {len(conversation_history)} turns")
                    continue
                
                elif user_input == '/help':
                    help_text = f"""
{Colors.CYAN}Available Commands:{Colors.END}

  {Colors.GREEN}/model <name>{Colors.END}     - Switch to checkpoint
  {Colors.GREEN}/models{Colors.END}            - List available models
  {Colors.GREEN}/temp <0.1-2.0>{Colors.END}   - Set temperature [current: {temperature}]
  {Colors.GREEN}/topk <1-100>{Colors.END}     - Set top-k [current: {top_k}]
  {Colors.GREEN}/topp <0.0-1.0>{Colors.END}   - Set top-p [current: {top_p}]
  {Colors.GREEN}/max <10-500>{Colors.END}     - Set max tokens [current: {max_tokens}]
  {Colors.GREEN}/reset{Colors.END}            - Clear conversation history
  {Colors.GREEN}/params{Colors.END}           - Show model statistics
  {Colors.GREEN}/help{Colors.END}             - Show this help
  {Colors.GREEN}/quit{Colors.END}             - Exit gracefully

Press Ctrl+C to exit at any time
"""
                    print(help_text)
                    continue
                
                elif user_input == '/models':
                    models = get_available_models()
                    if models:
                        model_list = "\n".join([f"  {Colors.GREEN}✦{Colors.END} {name}" for name in models])
                        print(f"{Colors.CYAN}Available models:{Colors.END}\n{model_list}")
                    else:
                        print(f"{Colors.YELLOW}No models found in checkpoints/ directory{Colors.END}")
                    continue
                
                elif user_input.startswith('/model'):
                    parts = user_input.split()
                    if len(parts) < 2:
                        print(f"{Colors.RED}✗ Usage: /model <name>{Colors.END}")
                        continue
                    model_name = parts[1]
                    model_path = Path("checkpoints") / f"{model_name}.pt"
                    if not model_path.exists():
                        print(f"{Colors.RED}✗ Model not found: {model_name}{Colors.END}")
                        continue
                    
                    print(f"\n{Colors.CYAN}Loading model: {model_name}...{Colors.END}")
                    try:
                        # Show loading spinner
                        for _ in range(2):
                            for spinner in SPINNER[:5]:
                                print(f"\r{Colors.CYAN}Loading{Colors.END} {spinner}", end='', flush=True)
                                time.sleep(0.05)
                        
                        model, tokenizer, config, step, loss = load_checkpoint(str(model_path), device)
                        current_checkpoint[0] = str(model_path)
                        conversation_history = []  # Clear history when switching models
                        
                        print(f"\r{Colors.GREEN}✓ Model switched to {model_name}!{Colors.END}")
                        print(f"{Colors.CYAN}  Step: {step} | Loss: {loss:.4f}{Colors.END}" if isinstance(loss, float) else f"{Colors.CYAN}  Step: {step}{Colors.END}")
                    except Exception as e:
                        print(f"\r{Colors.RED}✗ Failed to load model: {e}{Colors.END}")
                    continue
                
                elif user_input.startswith('/temp'):
                    parts = user_input.split()
                    if len(parts) == 2:
                        try:
                            new_temp = float(parts[1])
                            if 0.1 <= new_temp <= 2.0:
                                temperature = new_temp
                                print(f"{Colors.GREEN}✓ Temperature set to {temperature}{Colors.END}")
                            else:
                                print(f"{Colors.RED}✗ Temperature must be between 0.1 and 2.0{Colors.END}")
                        except ValueError:
                            print(f"{Colors.RED}✗ Invalid temperature value{Colors.END}")
                    else:
                        print(f"{Colors.CYAN}Current temperature: {temperature}{Colors.END}")
                    continue
                
                elif user_input.startswith('/topk'):
                    parts = user_input.split()
                    if len(parts) == 2:
                        try:
                            new_topk = int(parts[1])
                            if 1 <= new_topk <= 100:
                                top_k = new_topk
                                print(f"{Colors.GREEN}✓ Top-k set to {top_k}{Colors.END}")
                            else:
                                print(f"{Colors.RED}✗ Top-k must be between 1 and 100{Colors.END}")
                        except ValueError:
                            print(f"{Colors.RED}✗ Invalid top-k value{Colors.END}")
                    else:
                        print(f"{Colors.CYAN}Current top-k: {top_k}{Colors.END}")
                    continue
                
                elif user_input.startswith('/topp'):
                    parts = user_input.split()
                    if len(parts) == 2:
                        try:
                            new_topp = float(parts[1])
                            if 0.0 <= new_topp <= 1.0:
                                top_p = new_topp
                                print(f"{Colors.GREEN}✓ Top-p set to {top_p}{Colors.END}")
                            else:
                                print(f"{Colors.RED}✗ Top-p must be between 0.0 and 1.0{Colors.END}")
                        except ValueError:
                            print(f"{Colors.RED}✗ Invalid top-p value{Colors.END}")
                    else:
                        print(f"{Colors.CYAN}Current top-p: {top_p}{Colors.END}")
                    continue
                
                elif user_input.startswith('/max'):
                    parts = user_input.split()
                    if len(parts) == 2:
                        try:
                            new_max = int(parts[1])
                            if 10 <= new_max <= 500:
                                max_tokens = new_max
                                print(f"{Colors.GREEN}✓ Max tokens set to {max_tokens}{Colors.END}")
                            else:
                                print(f"{Colors.RED}✗ Max tokens must be between 10 and 500{Colors.END}")
                        except ValueError:
                            print(f"{Colors.RED}✗ Invalid max tokens value{Colors.END}")
                    else:
                        print(f"{Colors.CYAN}Current max tokens: {max_tokens}{Colors.END}")
                    continue
                
                else:
                    print(f"{Colors.RED}✗ Unknown command: {user_input}{Colors.END}")
                    print(f"  Try: /help for available commands")
                    continue
            
            # Check if model is loaded
            if model is None:
                print(f"{Colors.RED}✗ No model loaded.{Colors.END} Use {Colors.GREEN}/model <name>{Colors.END} to load one.")
                continue
            
            # Add to history
            conversation_history.append(("user", user_input))
            
            # Generate response - simple prompt completion, no Human/AI format
            print(f"\n{Colors.CYAN}AI{Colors.END} {Colors.YELLOW}➜{Colors.END} ", end='', flush=True)
            
            # Show thinking spinner
            for _ in range(3):
                for spinner in SPINNER[:5]:
                    print(f"\r{Colors.CYAN}AI{Colors.END} {Colors.YELLOW}➜{Colors.END} {spinner} Thinking...", end='', flush=True)
                    time.sleep(0.1)
            
            print(f"\r{Colors.CYAN}AI{Colors.END} {Colors.YELLOW}➜{Colors.END} ", end='', flush=True)
            
            # Generate response using model.generate()
            response = generate_response(
                model, tokenizer, user_input, device,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=max_tokens,
                show_thinking=True
            )
            
            # Add to history
            conversation_history.append(("ai", response))
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}👋 Interrupted. Goodbye!{Colors.END}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}✗ Error: {e}{Colors.END}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
