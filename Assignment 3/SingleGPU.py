from transformers import GPT2Tokenizer
import torch
from tqdm.notebook import tqdm

block_size = 1024
batch_size = 8

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
with open(r'C:\College\Projects\Contlo AI Assignment\Assignment 3\textdata.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokens = tokenizer(text, return_tensors='pt')
tokens = tokens['input_ids']
tokens = tokens.squeeze(dim=0)
n = int(0.9*len(tokens)) # first 90% will be train, rest val
train_data = tokens[:n]
val_data = tokens[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss(model, eval_iters:int=200, device:str='cuda'):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X,Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def TrainLoop(
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    total_iters:int=5000,
    eval_interval:int=200,
    device:str='cuda'
):
    model.to(device)
    for iter in tqdm(range(total_iters)):
        if iter % eval_interval == 0 or iter == total_iters - 1:
            losses = estimate_loss(model, 200)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')
        xb,yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()