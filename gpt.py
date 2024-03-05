import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(17)

# configs
batch_size = 64
block_size = 128
train_test_split = 0.9
device = 'mps'
lr = 3e-4
epochs = 5000
eval_iters = 500
n_layers = 5
n_embd = 128
n_heads = 8
dropout = 0.2
save = True # boolean for saving the model weights

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

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B x T x C
        q = self.query(x) # B x T x C
        # computing attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # B x T x T
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # B x T x T
        wei = F.softmax(wei, dim = -1) # B x T x T
        wei = self.dropout(wei)
        # weighted aggregation of values
        v = self.value(x) # B x T x C
        out = wei @ v # B x T x C
        return out

class MultiHeadAttenion(nn.Module):
    """Multiple heads of attention running parallely"""

    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """Feed forward network to ingest and compute all the information from multi-headed self-attention"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """A block of transformer: computation followed by communication"""

    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttenion(head_size, num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Bigram model
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets = None):
        # idx and targets are (B,T) tensors
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device = device)) # (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
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
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond) # logits of shape (B, T, C) since targets is None
            logits = logits[:,-1,:] # taking the latest time step to get (B, C)
            probs = torch.softmax(logits, dim = -1)
            next_idx = torch.multinomial(probs, num_samples = 1,replacement = True)
            idx = torch.cat((idx, next_idx), dim = -1) # make it (B, T+1)

        return idx
    
def train_model(optimizer, model):
    # training loop
    for i in range(epochs):
        # forward pass
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)

        # backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % eval_iters == 0:
            print(f'Loss at iteration {i+1} = {loss.item()}')

if __name__ == "__main__":
    model = GPTLanguageModel()
    m = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}\n")
    optimizer = torch.optim.Adam(m.parameters(), lr = lr)

    train_model(optimizer=optimizer, model=m)

    context = torch.zeros((1,1), dtype = torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens = 500)[0].tolist()))

    # Saving the model weights to a file
    if save == True:
        torch.save(model.state_dict(), 'model_weights.pth')

    #open('output.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))