from torch.utils.checkpoint import checkpoint as torch_checkpoint


def checkpoint(function, *args, **kwargs):
    return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)

