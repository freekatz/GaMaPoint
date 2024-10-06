import torch
import torch.nn.functional as F
from einops import rearrange
depths = torch.tensor([[0., 2., 3., 0.2], [1.1, 2., 0., 4.]])
print(depths.shape)
visible = (depths != 0).int()
print(visible, visible.shape)

