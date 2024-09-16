import torch
from einops import rearrange


def inverse_permutation(perm):
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.size(0), device=perm.device)
    return inv


class BaseOrder(object):
    def __init__(self):
        pass

    def sort(self, x):
        return x

    def inv_sort(self, x):
        return x


class Order(BaseOrder):
    def __init__(self, idx):
        super().__init__()
        assert len(idx.shape) == 2
        B, N = idx.shape
        self.B = B
        self.N = N

        self.idx_all = rearrange(idx, 'b n -> (b n)')
        # make order
        for i in range(1, B):
            self.idx_all[i * N:] = self.idx_all[i * N:] + N
        self.inv_idx_all = inverse_permutation(self.idx_all)

    def __apply_idx__(self, x, idx_all):
        assert x.shape[0] == self.B
        assert x.shape[1] == self.N

        shape_list = []
        for i in range(2, len(x.shape)):
            shape_list.append(chr(ord('a')+i))
        shape = ' '.join(shape_list)

        pattern = f'a b {shape} -> (a b) {shape}'
        x_all = rearrange(x, pattern)
        x_all = x_all[idx_all]
        pattern = f'(a b) {shape} -> a b {shape}'
        x = rearrange(x_all, pattern, a=self.B, b=self.N)
        return x

    def sort(self, x):
        """
        :param x: [B, N, *]
        :return: [B, N, *]
        """
        return self.__apply_idx__(x, self.idx_all)

    def inv_sort(self, x):
        """
        :param x: [B, N, *]
        :return: [B, N, *]
        """
        return self.__apply_idx__(x, self.inv_idx_all)


if __name__ == '__main__':
    B = 2
    N = 5
    C = 1
    p = torch.randn((B, N, 3))
    f = torch.randn((B, C, N))
    p[:, :, 2] = torch.tensor([[0, 2, 1, 4, 5], [1, 4, 5, 2, 3]])
    camid = p[:, :, 2]
    idx = torch.argsort(camid, dim=1, descending=True)

    f = f.permute(0, 2, 1)
    order = Order(idx)
    print(f)
    f = order.sort(f)
    print(f)
    f = order.inv_sort(f)
    print(f)
