import os
import torch
import torchvision
from torchvision import transforms

batch_size = 100
mnist_folder = os.path.join(os.path.dirname(__file__), 'mnist')
train_dataset = torchvision.datasets.MNIST(root=mnist_folder,
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=mnist_folder,
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
