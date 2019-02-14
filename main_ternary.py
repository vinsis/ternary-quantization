import os
import torch
import torch.nn as nn

from model import model_to_quantify, device
from data import train_loader, test_loader
from quantification import quantize, get_quantization_grads

criterion = nn.CrossEntropyLoss()
num_epochs = 2

# load model with full precision trained weights
dirname = os.path.dirname(__file__)
dirname = os.path.join(dirname, 'weights')
weightname = os.path.join(dirname, '{}.ckpt'.format('original'))
model_to_quantify.load_state_dict(torch.load(weightname, map_location='cpu'))

# create a list of parameters that need to be quantized
'''
Model parameter names and parameter sizes:
[('layer1.0.weight', torch.Size([16, 1, 5, 5])),
 ('layer1.0.bias', torch.Size([16])),
 ('layer1.1.weight', torch.Size([16])),
 ('layer1.1.bias', torch.Size([16])),
 ('layer2.0.weight', torch.Size([32, 16, 5, 5])),
 ('layer2.0.bias', torch.Size([32])),
 ('layer2.1.weight', torch.Size([32])),
 ('layer2.1.bias', torch.Size([32])),
 ('fc.weight', torch.Size([10, 1568])),
 ('fc.bias', torch.Size([10]))]

 layer1.1.* and layer2.1.* correspond to batch normalization layers.
 We do not quantize BN layers for now.
'''

bn_weights = [ param for name,param in model_to_quantify.named_parameters() if '.1' in name]
weights_to_be_quantized = [ param for name,param in model_to_quantify.named_parameters() if not '.1' in name]

# store a full precision copy of parameters that need to be quantized
full_precision_copies = [ param.data.clone().requires_grad_().to(device) for param in weights_to_be_quantized ]

# for each parameter to be quantized, create a trainable tensor of scaling factors (w_p and w_n)
# scaling_factors = torch.ones(len(weights_to_be_quantized), 2, requires_grad=True).to(device)
scaling_factors = [torch.ones(2, requires_grad=True).to(device) for _ in range(len(weights_to_be_quantized))]

# create optimizers for different parameter groups

# optimizer for the networks parameters containing quantized and batch norm weights
optimizer_main = torch.optim.Adam(
                    [{'params': bn_weights}, {'params': weights_to_be_quantized}],
                    lr=0.001
                )
# optimizers for full precision and scaling factors
optimizer_full_precision_weights = torch.optim.Adam(full_precision_copies, lr=0.0001)
optimizer_scaling_factors = torch.optim.Adam(scaling_factors, lr=0.0001)

def train():
    total_step = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        # quantize weights from full precision weights
        for index, weight in enumerate(weights_to_be_quantized):
            w_p, w_n = scaling_factors[index]
            weight.data = quantize(full_precision_copies[index].data, w_p, w_n)
        # forward pass
        images = images.to(device)
        labels = labels.to(device)

        outputs = model_to_quantify(images)
        loss = criterion(outputs, labels)

        # backward pass - calculate gradients
        optimizer_main.zero_grad()
        optimizer_full_precision_weights.zero_grad()
        optimizer_scaling_factors.zero_grad()
        loss.backward()

        for index, weight in enumerate(weights_to_be_quantized):
            w_p, w_n = scaling_factors[index]
            full_precision_data = full_precision_copies[index].data
            full_precision_grad, w_p_grad, w_n_grad = get_quantization_grads(weight.grad.data, full_precision_data, w_p.item(), w_n.item())
            full_precision_copies[index].grad = full_precision_grad.to(device)
            scaling_factors[index].grad = torch.FloatTensor([w_p_grad, w_n_grad]).to(device)
            weight.grad.data.zero_()

        if (i+1) % 10 == 0:
            print('Iteration {}, loss: {}'.format(i+1, loss.item()))
            test()

        optimizer_main.step()
        optimizer_full_precision_weights.step()
        optimizer_scaling_factors.step()


def test():
    model_to_quantify.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_to_quantify(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    assert full_precision_copies[0].requires_grad is True
    assert len(weights_to_be_quantized) == len(scaling_factors)
    assert len(weights_to_be_quantized) == len(full_precision_copies)
    train()
    print(scaling_factors)
