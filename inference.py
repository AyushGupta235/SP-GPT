import torch
from gpt import decode, GPTLanguageModel

# config
device = 'mps'
max_new_tokens = 10000

model = GPTLanguageModel()
model.load_state_dict(torch.load('model_weights.pth'))
m = model.to(device)
m.eval()

# Initialize context with a starting token
context = torch.zeros((1, 1), dtype=torch.long, device=device)
open(f'output_{max_new_tokens}.txt', 'w').write(decode(m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))