import os 
import random 

import torch 
import torch.nn as nn 
from torch.nn import functional as F 


# Hyperparameters 
batch_size = 32
block_size = 8 
max_iters = 3000
eval_interval = 1e-2
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 
n_embd = 32 
# -------------------



torch.manual_seed(1337)

filename = os.path.join("datasets", "input.txt")

# read and inspect 
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)


# create a mapping from characters to integers 
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers 
decode = lambda l: ''.join(itos[i] for i in l) # decoder: take a list of integers, output string 


# Train and test 
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val 
train_data = data[:n]
val_data = data[n:]


# data loading 
def get_batch(split):
    # generate a small batch of ata of inputs x and targets y 
    data = train_data if split == "train" else val_data 
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    X = torch.stack([data[i:i+block_size] for i in ix ])
    Y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    X, Y = X.to(device), Y.to(device)

    return X, Y 



class BigramLanguageModel(nn.Module): 

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a looup table 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape 
        # idx and targets are both (B, T) tensor of integers 
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T, C 
        x = token_emb * pos_emb # (B, T, C)
        logits = self.lm_head(token_emb) # (B, T, vocab_size) 


        if targets is None:
            loss = None 
        else:
            B, T, C = logits.shape 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits , loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context 
        for _ in range(max_new_tokens):
            # idx is (B, T) array of indices in the current context 
            logits, loss =  self(idx)
            # get the predictions 
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get probalilities 
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx 




@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out 



model = BigramLanguageModel(vocab_size)
m = model.to(device)


# create a PyTorch optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # evaluate loss on train and val sets 
    if iter % eval_interval ==0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


    # sample a batch of size 
    xb, yb = get_batch('train')

    # evaluate the loss 
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# generate from the model 
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))