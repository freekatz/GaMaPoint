from dataclasses import dataclass

import torch

from utils.dict_utils import ObjDict


@dataclass
class StructuredMask:
    def __init__(self, mask_type: str, mask_params: ObjDict):
        self.mask_type = mask_type
        self.mask_params = mask_params

    @classmethod
    def get_mask_fn(cls, mask_type, mask_params: ObjDict):
        fn = todo_fn()
        if mask_type == 'cov3d':
            fn = cov3d_fn(**mask_params)
        elif mask_type == 'cov2d':
            pass

        return fn

    def apply(self, x, is_post_mask=False):
        fn = self.get_mask_fn(self.mask_type, self.mask_params)
        return fn(x, is_post_mask=is_post_mask)


def todo_fn():
    def fn(x, is_post_mask=False):
        return x

    return fn


def cov3d_fn(cov3d, d_model):
    """
    :param cov3d: [B, N, 3, 3]
    :param d_model:
    """
    L, _ = torch.linalg.eigh(cov3d)
    L, _ = L.max(dim=-1, keepdim=False)  # [B, N]
    L = L.unsqueeze(-1)
    print(L)

    def fn(x, is_post_mask=False):
        if not is_post_mask:
            return x * L
        else:
            return x

    return fn


if __name__ == '__main__':
    cov3d = torch.randn(1, 2, 3, 3)
    mask = StructuredMask('cov3d', {'cov3d': cov3d, 'd_model': 3})

    x = torch.randn(1, 2, 3)
    x_old = x.clone()
    x = mask.apply(x, is_post_mask=False)
    print(x.shape)
    print(x)

    x = mask.apply(x, is_post_mask=True)
    print(x.shape)
    print(x)

    x_old = mask.apply(x_old, is_post_mask=True)
    print(x_old.shape)
    print(x_old)


