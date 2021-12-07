"""
Test LeNet5

Example:
python test_lenet5.py \
  --best-checkpoint ./checkpoints/LeNet5/epoch-6.pth \
  --device cpu
"""
import torch
import os
from torchvision import datasets, transforms
from models.YourNet import YourNet
from eval.metrics import get_accuracy, get_infer_time, get_macs_and_params

if __name__ == '__main__':
    batch_size = 64
    device = 'cuda'
    dictionary = './checkpoints/yournet_keepclear/'
    checkpoint_list = os.listdir(dictionary)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    the_accu = 0
    the_checkpoint = None
    for checkpoint in checkpoint_list:
        model = YourNet().to(device=device)
        try:
            model.load_state_dict(torch.load(dictionary + checkpoint, map_location=device))
        except RuntimeError:
            continue
        accuracy = get_accuracy(model, test_loader, device)
        print(checkpoint, accuracy)
        if accuracy > the_accu:
            the_accu = accuracy
            the_checkpoint = checkpoint
    print(the_checkpoint, the_accu)