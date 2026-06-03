from helper_functions import build_pretokenizer_pattern
import regex as re
import torch

x = torch.tensor([[1,2],[3,4]])
print(x.max(dim=1).values)