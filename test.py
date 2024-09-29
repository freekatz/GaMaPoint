import torch
import torch.nn.functional as F
from einops import rearrange

label = torch.tensor([[[0, 2, 3, 0], [1, 2, 0, 4]], [[3, 0, 0, 6], [0, 4, 5, 0]]])
label = rearrange(label, 'b n c -> c (b n)')
print(label, label.shape)
# one_hot = F.one_hot(label.long(), num_classes=4*2)
# print(one_hot, one_hot.shape)
# code = one_hot.sum(dim=1).float()
# code[:, 0] = 0
# print(code, code.shape)
i = torch.arange(1, 5)
print(i)
label = label*i.unsqueeze(1)
label = label.transpose(0, 1).float()
label = label.mean(dim=-1)
print(label)
