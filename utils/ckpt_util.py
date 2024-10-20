import logging
from collections import defaultdict, OrderedDict

import torch
from termcolor import colored


def resume_state(model, ckpt, compat=False, **args):
    state = torch.load(ckpt)
    if compat:  # compatible old model
        new_state_dict = OrderedDict()
        for key in list(state['model'].keys()):
            new_state_dict[key.replace('sub_stage', 'sub').replace('stage', 'backbone')] = state['model'].pop(key)
        state = new_state_dict
    model.load_state_dict(state['model'], strict=True)
    for i in args.keys():
        if hasattr(args[i], "load_state_dict"):
            args[i].load_state_dict(state[i])
    return state


def load_state(model, ckpt, compat=False, **args):
    state = torch.load(ckpt)
    if compat:  # compatible old model
        new_state_dict = OrderedDict()
        for key in list(state['model'].keys()):
            new_state_dict[key.replace('sub_stage', 'sub').replace('stage', 'backbone')] = state['model'].pop(key)
        state = new_state_dict
    if hasattr(model, 'module'):
        incompatible = model.module.load_state_dict(state['model'], strict=False)
    else:
        incompatible = model.load_state_dict(state['model'], strict=False)

    if incompatible.missing_keys:
        logging.info('missing_keys')
        logging.info(
            get_missing_parameters_message(incompatible.missing_keys),
        )
    if incompatible.unexpected_keys:
        logging.info('unexpected_keys')
        logging.info(
            get_unexpected_parameters_message(incompatible.unexpected_keys)
        )
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


def get_missing_parameters_message(keys):
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters or buffers are not found in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys):
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint state_dict contains keys that are not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta") for k, v in groups.items()
    )
    return msg


def _group_checkpoint_keys(keys):
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1:]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group):
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"
