import torch


def load_state(ckpt, **args):
    state = torch.load(ckpt)
    for i in args.keys():
        if hasattr(args[i], "load_state_dict"):
            args[i].load_state_dict(state[i])
    return state


def save_state(ckpt, **args):
    state = {}
    for i in args.keys():
        item = args[i].state_dict() if hasattr(args[i], "state_dict") else args[i]
        state[i] = item
    torch.save(state, ckpt)


def cal_model_params(model):
    total = sum([param.nelement() for param in model.parameters()])
    trainable = sum([param.nelement() for param in model.parameters() if param.requires_grad])

    return total, trainable

