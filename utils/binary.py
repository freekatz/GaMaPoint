import torch


def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


if __name__ == '__main__':
    bin1 = torch.tensor([[1, 0, 0, 1], [1, 0, 0, 1]])
    bin2 = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]])
    i1 = bin2dec(bin1, 4)
    i2 = bin2dec(bin2, 4)
    i = i1 & i2
    bin = dec2bin(i, 4)
    print(i, bin)
    print(i.shape, bin.shape)
