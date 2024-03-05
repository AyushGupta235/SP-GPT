import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(17)

# configs
batch_size = 32
block_size = 8
train_test_split = 0.9
device = 'mps'
lr = 1e-3
epochs = 50000
eval_iters = 5000
n_embd = 32

# loading the dataset
#wget https://github.com/karpathy/ng-video-lecture/blob/master/input.txt

with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)
n = int(train_test_split * len(text)) # train-test split
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets = None):
        # idx and targets are (B,T) tensors
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + pos_emb
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # logits of shape (B, T, C) since targets is None
            logits = logits[:,-1,:] # taking the latest time step to get (B, C)
            probs = torch.softmax(logits, dim = -1)
            next_idx = torch.multinomial(probs, num_samples = 1,replacement = True)
            idx = torch.cat((idx, next_idx), dim = -1) # make it (B, T+1)

        return idx
    
model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.Adam(m.parameters(), lr = lr)

# training loop
for i in range(epochs):
    # forward pass
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)

    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if i % eval_iters == 0:
        print(f'Loss at iteration {i+1} = {loss.item()}')

idx = torch.zeros((1,1), dtype = torch.long, device=device)
print(decode(m.generate(idx, max_new_tokens = 1000)[0].tolist()))