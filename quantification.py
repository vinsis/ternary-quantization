import torch

def quantize(tensor_data, w_p, w_n, threshold=0.15):
    """
    Returns quantized weights of tensor_data.
    w_p, w_n, threshold are all learnt
    Set of possible weights: {-w_n, 0, w_p}
    """
    delta = tensor_data.abs().max() * threshold
    return (tensor_data > delta).float() * w_p + (tensor_data < -delta).float() * -w_n
