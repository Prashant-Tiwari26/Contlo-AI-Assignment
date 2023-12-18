import torch
from SingleGPU import TrainLoop
from GPT2Modified import GPT2

model = GPT2()
model.from_pretrained()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

TrainLoop(model, optimizer)