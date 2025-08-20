import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

# Load and preprocess the dataset
with open("Hyprog_LLM/dataset/Hobbit.txt", "r", encoding="utf-8") as f:
    raw_hobbit = f.read()

# Tokenize using GPT-2 tokenizer (BPE)
tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_hobbit)
print(f"Number of tokens: {len(enc_text)}")

# Sample demonstration of token context
context_size = 4
enc_sample = enc_text[50:54+context_size]

print("Token context demonstration:")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"{tokenizer.decode(context)} ----> {tokenizer.decode([desired])}")

# Dataset class for creating input-target pairs
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        # Use a sliding window to chunk the text into overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# DataLoader creation function
def create_dataloader(txt, batch_size=4, max_length=256, 
                      stride=128, shuffle=True, drop_last=True):
    # Create dataset
    dataset = GPTDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    return dataloader

# Create a small dataloader for testing
max_length = 4
dataloader = create_dataloader(
    raw_hobbit, batch_size=8, max_length=max_length,
    stride=max_length, shuffle=False
)

# Test the dataloader
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("\nInputs shape:", inputs.shape)
print("Targets shape:", targets.shape)

# Initialize embedding layers
vocab_size = tokenizer.n_vocab  # Use the actual vocab size from GPT-2 tokenizer
output_dim = 256

# Token embedding layer
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
print("\nToken embeddings shape:", token_embeddings.shape)

# Positional embedding layer
pos_embedding_layer = torch.nn.Embedding(max_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print("Position embeddings shape:", pos_embeddings.shape)

# Combine token and position embeddings
input_embeddings = token_embeddings + pos_embeddings
print("Combined embeddings shape:", input_embeddings.shape)


# develop self attention...