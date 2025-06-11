import torch
import torch.nn as nn
import math
from transformers import RobertaTokenizer

# Constants
MAX_SEQ_LENGTH = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_hidden, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, ff_hidden, dropout, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_hidden, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, mask)
        x = x.transpose(0, 1)
        return self.fc_out(x)

# Prediction Function
def generate_Transformer(model, input_text,tokenizer = tokenizer,  max_length=100):
    model.eval()
    tokens = tokenizer.tokenize(input_text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = input_ids[:MAX_SEQ_LENGTH-1]
    input_tensor = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_tensor)
            next_token_logits = output[:, -1, :]
            next_token_logits = next_token_logits / 0.8  # temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    generated_ids = input_tensor[0].cpu().numpy()
    return tokenizer.decode(generated_ids[len(input_ids):], skip_special_tokens=True)

# Model Configuration (same as training)
vocab_size = tokenizer.vocab_size
d_model = 768
num_layers = 6
num_heads = 8
ff_hidden = 2048
dropout = 0.1
max_len = MAX_SEQ_LENGTH

# Instantiate and load model
modelTransformer = Transformer(vocab_size, d_model, num_layers, num_heads, ff_hidden, dropout, max_len)
modelTransformer.load_state_dict(torch.load("Models\Transformer200_model.pth", map_location=device))
modelTransformer.to(device)
print("[TRANSFORMER] Model loaded successfully!")

if __name__ == "__main__":
    input_prompt = "import numpy"
    generated_code = generate_Transformer(modelTransformer, input_prompt,tokenizer, max_length=50)
    print(generated_code)