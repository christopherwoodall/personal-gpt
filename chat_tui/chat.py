#!/usr/bin/env python3
"""
Gothic Dandy Chat TUI
Kawaii Cyberpunk Terminal Interface
Single-file implementation using Textual
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Input, RichLog, Select, Static
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import ModelConfig
from src.model import GPT


class GothicChat(App):
    """
    Kawaii Cyberpunk Chat Interface
    
    Slash Commands:
        /temp <float>     - Set temperature (0.1-2.0)
        /topk <int>       - Set top-k (1-100)
        /topp <float>     - Set top-p (0.0-1.0)
        /max <int>        - Set max tokens (10-500)
        /model <name>     - Switch model
        /models           - List available models
        /clear            - Clear chat history
        /save             - Save chat to logs/
        /help             - Show commands
    """
    
    CSS = """
    /* ╭──────────────────────────────────────────╮
       │     Kawaii Cyberpunk Theme              │
       ╰──────────────────────────────────────────╯ */
    
    Screen {
        background: #0a0a0f;
    }
    
    /* Header */
    #header {
        background: #1a1a2e;
        color: #ff6ac1;
        text-style: bold;
        height: 1;
        content-align: center middle;
        border-bottom: solid #ff6ac1;
    }
    
    /* Model Selector */
    #model-select {
        background: #16213e;
        color: #00f5ff;
        border: solid #bf7af0;
        height: 3;
        margin: 0 1;
    }
    
    /* Chat Container */
    #chat-container {
        border: solid #ff6ac1;
        background: #0f0f1a;
        height: 1fr;
        margin: 1;
        padding: 0;
    }
    
    /* Chat Log */
    #chat-log {
        background: #0f0f1a;
        color: #e0e0e0;
        padding: 1;
        border: none;
    }
    
    /* Input Area */
    #input-container {
        height: 3;
        margin: 0 1 1 1;
    }
    
    #chat-input {
        background: #1a1a2e;
        color: #00f5ff;
        border: tall #ff6ac1;
    }
    
    #chat-input:focus {
        border: tall #00f5ff;
    }
    
    /* Status Bar */
    #status-bar {
        background: #16213e;
        color: #bf7af0;
        height: 1;
        content-align: left middle;
        padding: 0 1;
    }
    
    /* Scrollbar styling */
    RichLog > .scrollbar {
        background: #1a1a2e;
    }
    
    RichLog:focus > .scrollbar {
        background: #ff6ac1;
    }
    """
    
    # Reactive state
    temperature = reactive(0.8)
    top_k = reactive(40)
    top_p = reactive(0.9)
    max_tokens = reactive(200)
    current_model = reactive("")
    is_generating = reactive(False)
    
    def __init__(self):
        super().__init__()
        self.model: Optional[GPT] = None
        self.tokenizer: Optional[Tokenizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.chat_history: List[dict] = []
        self.checkpoints_dir = Path("checkpoints")
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
    def compose(self) -> ComposeResult:
        """Build the UI layout"""
        # Header
        yield Static("✦ Gothic Dandy Chat ✦  Type /help for commands", id="header")
        
        # Model selector
        models = self._get_available_models()
        yield Select(models, id="model-select", prompt="Select Model ✧")
        
        # Chat area
        with Vertical(id="chat-container"):
            yield RichLog(id="chat-log", highlight=True, wrap=True)
        
        # Input area
        with Horizontal(id="input-container"):
            yield Input(
                placeholder="✧ Type your message or /command...",
                id="chat-input"
            )
        
        # Status bar
        yield Static(
            f"Device: {self.device} | Ready ✦",
            id="status-bar"
        )
    
    def _get_available_models(self) -> List[tuple]:
        """Get list of available checkpoint files"""
        models = [("None", "")]
        
        if self.checkpoints_dir.exists():
            for ckpt in sorted(self.checkpoints_dir.glob("*.pt")):
                models.append((ckpt.stem, str(ckpt)))
        
        return models
    
    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle model selection"""
        if event.value and event.value != "":
            await self._load_model(event.value)
    
    async def _load_model(self, checkpoint_path: str) -> None:
        """Load a model checkpoint"""
        self._add_system_message(f"Loading model: {Path(checkpoint_path).name} ✧")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Get config from checkpoint or use default
            if "config" in checkpoint:
                config = checkpoint["config"]
            else:
                config = ModelConfig()
            
            # Load tokenizer
            tokenizer_path = Path("data/tokenizer.json")
            if not tokenizer_path.exists():
                self._add_error_message("Tokenizer not found! Run: gothic-tokenize")
                return
            
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            config.vocab_size = self.tokenizer.get_vocab_size()
            
            # Create and load model
            self.model = GPT(config)
            self.model.load_state_dict(checkpoint["model"])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.current_model = Path(checkpoint_path).stem
            step = checkpoint.get("step", "unknown")
            loss = checkpoint.get("best_val_loss", "unknown")
            
            self._add_system_message(
                f"✦ Model loaded! Step: {step} | Loss: {loss:.4f}" 
                if isinstance(loss, float) else f"✦ Model loaded! Step: {step}"
            )
            self._update_status(f"Model: {self.current_model} | Ready ✦")
            
        except Exception as e:
            self._add_error_message(f"Failed to load model: {e}")
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input"""
        message = event.value.strip()
        if not message:
            return
        
        # Clear input
        self.query_one("#chat-input", Input).value = ""
        
        # Handle slash commands
        if message.startswith("/"):
            await self._handle_command(message)
            return
        
        # Generate response
        if self.model is None:
            self._add_error_message("Please select a model first! ✧")
            return
        
        await self._generate_response(message)
    
    async def _handle_command(self, command: str) -> None:
        """Process slash commands"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "/help":
            help_text = """
✦ Gothic Dandy Chat Commands ✦

/model <name>     - Switch to checkpoint
/models           - List available models
/temp <0.1-2.0>   - Set temperature [current: {temp}]
/topk <1-100>     - Set top-k [current: {topk}]
/topp <0.0-1.0>   - Set top-p [current: {topp}]
/max <10-500>     - Set max tokens [current: {max}]
/clear            - Clear chat history
/save             - Save chat to logs/
/help             - Show this help

Press Ctrl+C to exit
"".format(
                temp=self.temperature,
                topk=self.top_k,
                topp=self.top_p,
                max=self.max_tokens
            )
            self._add_system_message(help_text)
        
        elif cmd == "/model":
            if len(parts) < 2:
                self._add_error_message("Usage: /model <checkpoint_name>")
                return
            model_name = parts[1]
            model_path = self.checkpoints_dir / f"{model_name}.pt"
            if model_path.exists():
                await self._load_model(str(model_path))
            else:
                self._add_error_message(f"Model not found: {model_name}")
        
        elif cmd == "/models":
            models = self._get_available_models()
            model_list = "\n".join([f"  ✦ {name}" for name, _ in models if name != "None"])
            self._add_system_message(f"Available models:\n{model_list}")
        
        elif cmd == "/temp":
            if len(parts) < 2:
                self._add_system_message(f"Current temperature: {self.temperature}")
                return
            try:
                temp = float(parts[1])
                if 0.1 <= temp <= 2.0:
                    self.temperature = temp
                    self._add_system_message(f"✦ Temperature set to: {temp}")
                else:
                    self._add_error_message("Temperature must be between 0.1 and 2.0")
            except ValueError:
                self._add_error_message("Invalid temperature value")
        
        elif cmd == "/topk":
            if len(parts) < 2:
                self._add_system_message(f"Current top-k: {self.top_k}")
                return
            try:
                topk = int(parts[1])
                if 1 <= topk <= 100:
                    self.top_k = topk
                    self._add_system_message(f"✦ Top-k set to: {topk}")
                else:
                    self._add_error_message("Top-k must be between 1 and 100")
            except ValueError:
                self._add_error_message("Invalid top-k value")
        
        elif cmd == "/topp":
            if len(parts) < 2:
                self._add_system_message(f"Current top-p: {self.top_p}")
                return
            try:
                topp = float(parts[1])
                if 0.0 <= topp <= 1.0:
                    self.top_p = topp
                    self._add_system_message(f"✦ Top-p set to: {topp}")
                else:
                    self._add_error_message("Top-p must be between 0.0 and 1.0")
            except ValueError:
                self._add_error_message("Invalid top-p value")
        
        elif cmd == "/max":
            if len(parts) < 2:
                self._add_system_message(f"Current max tokens: {self.max_tokens}")
                return
            try:
                max_tokens = int(parts[1])
                if 10 <= max_tokens <= 500:
                    self.max_tokens = max_tokens
                    self._add_system_message(f"✦ Max tokens set to: {max_tokens}")
                else:
                    self._add_error_message("Max tokens must be between 10 and 500")
            except ValueError:
                self._add_error_message("Invalid max tokens value")
        
        elif cmd == "/clear":
            self.chat_history.clear()
            chat_log = self.query_one("#chat-log", RichLog)
            chat_log.clear()
            self._add_system_message("✦ Chat history cleared")
        
        elif cmd == "/save":
            await self._save_chat()
        
        else:
            self._add_error_message(f"Unknown command: {cmd}. Type /help for commands.")
    
    async def _generate_response(self, prompt: str) -> None:
        """Generate streaming response from model"""
        # Add user message
        self._add_user_message(prompt)
        
        self.is_generating = True
        self._update_status(f"Generating... ✦✧✦ (temp={self.temperature}, topk={self.top_k})")
        
        try:
            # Encode prompt
            encoded = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
            
            # Generate tokens
            generated_ids = input_ids[0].tolist()
            response_text = ""
            
            chat_log = self.query_one("#chat-log", RichLog)
            
            # Add assistant prefix
            self._add_assistant_prefix()
            
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for _ in range(self.max_tokens):
                        # Prepare input
                        input_tensor = torch.tensor(
                            [generated_ids[-self.model.config.block_size:]],
                            dtype=torch.long,
                            device=self.device
                        )
                        
                        # Forward pass
                        logits, _ = self.model(input_tensor)
                        logits = logits[:, -1, :] / self.temperature
                        
                        # Apply top-k
                        if self.top_k > 0:
                            v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
                            logits[logits < v[:, [-1]]] = float("-inf")
                        
                        # Apply top-p
                        if self.top_p > 0.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(
                                torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
                            )
                            sorted_indices_to_remove = cumulative_probs > self.top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(
                                1, sorted_indices, sorted_indices_to_remove
                            )
                            logits[indices_to_remove] = float("-inf")
                        
                        # Sample
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Check for EOS
                        if next_token.item() == self.tokenizer.token_to_id("<|endoftext|>"):
                            break
                        
                        generated_ids.append(next_token.item())
                        
                        # Decode and display
                        new_text = self.tokenizer.decode([next_token.item()])
                        response_text += new_text
                        
                        # Stream update (append to last line)
                        self._update_last_line(f"✦ {response_text}")
                        
                        # Small delay for visual effect
                        await asyncio.sleep(0.01)
            
            # Add to history
            self.chat_history.append({"role": "assistant", "content": response_text})
            
            self._update_status(f"Model: {self.current_model} | Ready ✦")
            
        except Exception as e:
            self._add_error_message(f"Generation error: {e}")
        
        finally:
            self.is_generating = False
    
    def _add_user_message(self, text: str) -> None:
        """Add user message to chat"""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write("")
        chat_log.write(Text(f"You ✧", style="bold #ff6ac1"))
        chat_log.write(Text(f"  {text}", style="#e0e0e0"))
        self.chat_history.append({"role": "user", "content": text})
    
    def _add_assistant_prefix(self) -> None:
        """Add assistant message prefix"""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write("")
        chat_log.write(Text(f"Assistant ✦", style="bold #00f5ff"))
    
    def _update_last_line(self, text: str) -> None:
        """Update the last line of chat (for streaming)"""
        chat_log = self.query_one("#chat-log", RichLog)
        # Remove last line and rewrite
        lines = chat_log.lines.copy()
        if lines:
            lines[-1] = Text(text, style="#00f5ff")
            chat_log.clear()
            for line in lines:
                chat_log.write(line)
    
    def _add_system_message(self, text: str) -> None:
        """Add system/info message"""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write("")
        chat_log.write(Text(text, style="#bf7af0"))
    
    def _add_error_message(self, text: str) -> None:
        """Add error message"""
        chat_log = self.query_one("#chat-log", RichLog)
        chat_log.write("")
        chat_log.write(Text(f"✦ Error: {text}", style="bold #ff6ac1"))
    
    def _update_status(self, text: str) -> None:
        """Update status bar"""
        status = self.query_one("#status-bar", Static)
        status.update(text)
    
    async def _save_chat(self) -> None:
        """Save chat history to logs directory"""
        if not self.chat_history:
            self._add_error_message("No chat history to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.logs_dir / f"chat_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "model": self.current_model,
            "settings": {
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens
            },
            "messages": self.chat_history
        }
        
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        
        self._add_system_message(f"✦ Chat saved to: {filename}")


if __name__ == "__main__":
    app = GothicChat()
    app.run()
