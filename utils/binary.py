import torch


def dec2bin(x, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)


def bin2dec_split(b, max_bits=64):
    bits = b.shape[-1]
    assert bits % max_bits == 0
    b = torch.split(b, max_bits, dim=-1)
    masks = []
    for _b in b:
        masks.append(bin2dec(_b, max_bits))
    return torch.stack(masks, dim=-1)


if __name__ == '__main__':
    b = (torch.randn(10, 8) > 0).to(torch.uint8)
    print(b)
    print(b.shape)
    i = bin2dec_split(b, 4)
    print(i)
    print(i.shape)

