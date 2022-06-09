# utilities for CVP-MVSNet
# The code is borrowed from the following github repository:
#       https://github.com/JiayuYANG/CVP-MVSNet

import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn.functional as F

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, float):
        return torch.tensor(vars).cuda()
    elif isinstance(vars, np.ndarray):
        return torch.tensor(vars).cuda()
    elif isinstance(vars, int):
        return torch.tensor(vars).cuda()
    else:
        raise NotImplementedError("invalid input type {}".format(type(vars)))
