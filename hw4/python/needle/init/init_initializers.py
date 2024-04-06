import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    if shape is not None:
        fan_in = math.prod(shape[:-1])
    else:
        shape = (fan_in, fan_out)
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(*shape, low = -a, high = a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    if shape is not None:
        fan_in = math.prod(shape[:-1])
    else:
        shape = (fan_in, fan_out)
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(*shape, mean = 0, std = std, **kwargs)
    ### END YOUR SOLUTION



def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is not None:
        fan_in = math.prod(shape[:-1])
    else:
        shape = (fan_in, fan_out)
    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    return rand(*shape, low = -bound, high = bound, **kwargs)
    ### END YOUR SOLUTION

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is not None:
        fan_in = math.prod(shape[:-1])
    else:
        shape = (fan_in, fan_out)
    std = math.sqrt(2) / math.sqrt(fan_in)
    return randn(*shape, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION