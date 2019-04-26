import os
import torch
import torch.nn as nn

from model import model_auto, device
from data import train_loader, test_loader

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_auto.parameters(), lr=1.0)
num_epochs = 2

def quantize_params(model = model_auto):
    for n,p in model.named_parameters():
        p.data = torch.sign(p.data) * 0.01

# def update_weights(model = model_auto):
#     for n,p in model.named_parameters():
#         p.data = p.grad.data * 0.1

def train(model = model_auto):
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            for param in optimizer.param_groups[0]['params']:
                param.grad.data = torch.sign(param.grad.data) * 0.001
            optimizer.step()

            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            if (i+1) % 100 == 0:
                test()
        test()

def test(model = model_auto):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

def save_model(model = model_auto):
    dirname = os.path.dirname(__file__)
    dirname = os.path.join(dirname, 'weights')
    weightname = os.path.join(dirname, '{}.ckpt'.format(model.name))
    torch.save(model.state_dict(), weightname)

if __name__ == '__main__':
    quantize_params()
    train()
    # test()
    # save_model()
