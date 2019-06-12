import torch

THRESHOLD = 0.15
def quantize(tensor_data, w_p, w_n, threshold=THRESHOLD):
    delta = tensor_data.abs().max() * threshold
    return (tensor_data > delta).float() * w_p + (tensor_data < -delta).float() * -w_n

def get_quantization_grads(grad_data, full_precision_data, w_p_data, w_n_data, threshold=THRESHOLD):
    delta = full_precision_data.abs().max() * threshold
    a = (full_precision_data > delta).float()
    b = (full_precision_data < -delta).float()
    c = torch.ones_like(full_precision_data) - a - b

    full_precision_grad = a * grad_data * w_p_data + b * grad_data * w_n_data + c * grad_data * 1
    w_p_grad = (a * grad_data).mean()
    w_n_grad = (b * grad_data).mean()
    return full_precision_grad, w_p_grad, w_n_grad
