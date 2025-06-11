import torch
import torch.nn as nn
from transformers import RobertaTokenizer
import numpy as np

# Load CodeBERT Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

class CodeLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super(CodeLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

#-----------------LSTM--------------------------
vocab_size = 50265
modelLSTM = CodeLSTM(vocab_size)
modelLSTM.load_state_dict(torch.load("Models\code_lstm100pre_model.pth"))
print("[LSTM] Model loaded successfully!")

def generate_codeLSTM(modelLSTM , seed_text, max_length=10):
    modelLSTM.eval()
    input_tokens = tokenizer.tokenize(seed_text)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_ids = torch.tensor([input_ids])

    generated_tokens = []
    for _ in range(max_length):
        with torch.no_grad():
            output = modelLSTM(input_ids)
            next_token_id = torch.argmax(output[:, -1, :]).item()
            if next_token_id == tokenizer.pad_token_id:
                break
            generated_tokens.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)

    return tokenizer.decode(generated_tokens)

#----------------GRU----------------------------
# Define GRU Model
class CodeGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super(CodeGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=tokenizer.pad_token_id)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out)
        return output

modelGRU = CodeGRU(vocab_size)
modelGRU.load_state_dict(torch.load("Models\code_GRU100pre_model.pth"))
print("[GRU] Model loaded successfully!")

def generate_codeGRU(modelGRU , seed_text, max_length=10):
    print("Called")
    modelGRU.eval()
    input_tokens = tokenizer.tokenize(seed_text)
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    input_ids = torch.tensor([input_ids])

    generated_tokens = []
    for _ in range(max_length):
        with torch.no_grad():
            output = modelGRU(input_ids)
            next_token_id = torch.argmax(output[:, -1, :]).item()
            if next_token_id == tokenizer.pad_token_id:
                break
            generated_tokens.append(next_token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)

    return tokenizer.decode(generated_tokens)



if __name__ == "__main__":
    print(generate_codeLSTM(modelLSTM , "import numpy"))
    
