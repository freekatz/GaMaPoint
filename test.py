import torch
import torch.nn.functional as F
from einops import rearrange, repeat

camid = torch.tensor([[[0, 2, 3, 0]], [[1, 0, 0, 4]]], dtype=torch.float)
print(camid.shape)
n_cameras = 2
i = torch.arange(1, n_cameras*2+1)
i = repeat(i, 'c -> n d c', n=camid.shape[0], d=1)
print(camid.mean(dim=-1))
camid = camid * i
print(camid.mean(dim=-1))
