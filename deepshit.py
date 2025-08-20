# tolkien_llm.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import json
from pathlib import Path
import math
import time
from typing import List, Dict, Tuple

# ==================== CONFIGURATION ====================
class Config:
    # Model architecture
    vocab_size = 10000  # Will be updated based on actual vocabulary
    n_embd = 256        # Embedding dimension
    n_head = 4          # Number of attention heads
    n_layer = 4         # Number of transformer layers
    block_size = 128    # Context length
    dropout = 0.1
    
    # Training
    batch_size = 16
    learning_rate = 5e-4
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200
    
    # Data
    data_dir = "Hyprog_LLM/dataset"
    model_save_path = "tolkien_lm.pth"
    vocab_file = "vocab.json"

# ==================== DATA PROCESSING ====================
class TolkienDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks
        
    def __len__(self):
        return len(self.text_chunks)
    
    def __getitem__(self, idx):
        chunk = self.text_chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class TolkienDataProcessor:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0
        
    def load_and_clean_text(self, file_pattern="*.txt"):
        data_dir = Path(Config.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Directory {Config.data_dir} not found. "
                                  "Please create it and add Tolkien text files.")
        
        text_files = list(data_dir.glob(file_pattern))
        if not text_files:
            raise FileNotFoundError(f"No text files found in {Config.data_dir}")
        
        full_text = ""
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text += f.read() + " "
        
        # Clean text
        full_text = re.sub(r'\s+', ' ', full_text)  # Replace multiple spaces
        full_text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', '', full_text)  # Remove special chars
        return full_text
    
    def build_vocabulary(self, text, max_vocab_size=10000):
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Sort by frequency and take top tokens
        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
        vocab = {char: i+2 for i, (char, count) in enumerate(sorted_chars[:max_vocab_size-2])}
        
        # Add special tokens
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
        Config.vocab_size = self.vocab_size
        
        # Save vocabulary
        with open(Config.vocab_file, 'w') as f:
            json.dump(vocab, f)
            
        return vocab
    
    def encode_text(self, text):
        encoded = []
        for char in text:
            encoded.append(self.vocab.get(char, self.vocab['<UNK>']))
        return encoded
    
    def decode_tokens(self, tokens):
        return ''.join([self.inverse_vocab.get(token, '') for token in tokens])
    
    def prepare_datasets(self, text, train_ratio=0.9):
        # Encode text
        encoded_text = self.encode_text(text)
        
        # Split into training and validation
        n = len(encoded_text)
        train_data = encoded_text[:int(n * train_ratio)]
        val_data = encoded_text[int(n * train_ratio):]
        
        # Create chunks of block_size
        train_chunks = []
        for i in range(0, len(train_data) - Config.block_size, Config.block_size):
            train_chunks.append(train_data[i:i+Config.block_size+1])
            
        val_chunks = []
        for i in range(0, len(val_data) - Config.block_size, Config.block_size):
            val_chunks.append(val_data[i:i+Config.block_size+1])
        
        return train_chunks, val_chunks

# ==================== MODEL ARCHITECTURE ====================
class Head(nn.Module):
    """One self-attention head"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(Config.block_size, Config.block_size)))
        self.dropout = nn.Dropout(Config.dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores
        weights = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = torch.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        
        # Perform weighted aggregation
        v = self.value(x)
        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)
        self.dropout = nn.Dropout(Config.dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """Simple linear layer followed by non-linearity"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.n_embd, 4 * Config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * Config.n_embd, Config.n_embd),
            nn.Dropout(Config.dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""
    def __init__(self):
        super().__init__()
        head_size = Config.n_embd // Config.n_head
        self.sa = MultiHeadAttention(Config.n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(Config.n_embd)
        self.ln2 = nn.LayerNorm(Config.n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TolkienLM(nn.Module):
    """Language model for Tolkien's works"""
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(Config.vocab_size, Config.n_embd)
        self.position_embedding = nn.Embedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, Config.vocab_size)
        
        # Weight sharing optimization
        self.token_embedding.weight = self.lm_head.weight
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get token and position embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Apply transformer blocks
        x = self.blocks(x)  # (B, T, n_embd)
        x = self.ln_f(x)    # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text given a context"""
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -Config.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# ==================== TRAINING AND EVALUATION ====================
def estimate_loss(model, train_loader, val_loader):
    """Estimate loss on training and validation sets"""
    model.eval()
    losses = {}
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        total_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i >= Config.eval_iters:
                    break
                x, y = x.to(device), y.to(device)
                _, loss = model(x, y)
                total_loss += loss.item()
        losses[split] = total_loss / min(len(loader), Config.eval_iters)
    
    model.train()
    return losses

def train_model(model, train_loader, val_loader, optimizer):
    """Train the language model"""
    model.train()
    train_losses, val_losses = [], []
    
    for iter in range(Config.max_iters):
        # Get a batch
        xb, yb = next(iter(train_loader))
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward pass
        _, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # Track stats
        if iter % 100 == 0:
            print(f"Iteration {iter}: loss = {loss.item():.4f}")
        
        # Evaluate
        if iter % Config.eval_interval == 0 or iter == Config.max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader)
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    return train_losses, val_losses

# ==================== QA SYSTEM ====================
class TolkienQA:
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor
        self.knowledge_base = self._build_knowledge_base()
        
    def _build_knowledge_base(self):
        """Extract key information from the text"""
        # This would be expanded to extract entities, relationships, etc.
        # For now, we'll just return a placeholder
        return {
            "characters": ["Frodo", "Gandalf", "Bilbo", "Aragorn", "Legolas", "Gimli", "Sauron"],
            "places": ["Shire", "Rivendell", "Mordor", "Gondor", "Rohan", "Moria"],
            "objects": ["Ring", "Sting", "Andúril", "Palantír"]
        }
    
    def answer_question(self, question, max_length=200):
        """Generate an answer to a question about Tolkien's works"""
        # Create context with question
        context = f"Question: {question}\nAnswer:"
        encoded = self.data_processor.encode_text(context)
        
        # Convert to tensor
        context_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
        
        # Generate response
        with torch.no_grad():
            generated = self.model.generate(context_tensor, max_new_tokens=max_length, temperature=0.8)
        
        # Decode response
        response_tokens = generated[0].tolist()
        response = self.data_processor.decode_tokens(response_tokens)
        
        # Extract just the answer part
        if "Answer:" in response:
            response = response.split("Answer:")[1]
        
        return response.strip()

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Step 1: Process data
    print("Loading and processing Tolkien texts...")
    processor = TolkienDataProcessor()
    
    try:
        text = processor.load_and_clean_text()
        print(f"Loaded text with {len(text)} characters")
        
        # Build vocabulary
        vocab = processor.build_vocabulary(text)
        print(f"Built vocabulary with {len(vocab)} tokens")
        
        # Prepare datasets
        train_chunks, val_chunks = processor.prepare_datasets(text)
        print(f"Prepared {len(train_chunks)} training chunks and {len(val_chunks)} validation chunks")
        
        # Create data loaders
        train_dataset = TolkienDataset(train_chunks)
        val_dataset = TolkienDataset(val_chunks)
        
        train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
        
        # Step 2: Initialize model
        print("Initializing model...")
        model = TolkienLM().to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
        
        # Step 3: Train model
        print("Training model...")
        optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)
        
        train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer)
        
        # Step 4: Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab': processor.vocab,
            'config': {k: v for k, v in vars(Config).items() if not k.startswith('_')}
        }, Config.model_save_path)
        print(f"Model saved to {Config.model_save_path}")
        
        # Step 5: Initialize QA system
        qa_system = TolkienQA(model, processor)
        
        # Step 6: Example questions
        questions = [
            "Who is Frodo Baggins?",
            "What is the One Ring?",
            "Tell me about Gandalf",
            "What is the Shire?",
            "Who are the hobbits?"
        ]
        
        print("\n=== Tolkien Lore QA System ===")
        for question in questions:
            print(f"\nQ: {question}")
            answer = qa_system.answer_question(question)
            print(f"A: {answer}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure you have Tolkien text files in the 'tolkien_data' directory.")