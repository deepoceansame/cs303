import argparse
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from models.LeNet5 import LeNet5
from models.YourNet import YourNet
from eval.metrics import get_accuracy

# arguments
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-dir', type=str, required=True)
parser.add_argument('--last-checkpoint', type=str, default=None)

parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, required=True)

args = parser.parse_args()


# 损失函数
def get_loss(t_logits, s_logits, label, a, T):
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss()
    loss = a * loss1(s_logits, label) + T * loss2(t_logits, s_logits)
    # print(loss1(s_logits, label),loss2(t_logits, s_logits))
    return loss


def teacher_predict(model, loader, device):
    model.eval()
    t_logits = []
    with torch.no_grad():
        for x, y in loader:
            # print(texts)
            x = x.to(device)
            outputs = model(x)
            t_logits.append(outputs)
    return t_logits


def train_student(T_model, S_model, train_loader, test_loader):
    start_epoch = args.epoch_start
    end_epoch = args.epoch_end
    device = args.device
    t_train_logits = teacher_predict(T_model, train_loader, device)
    optimizer = torch.optim.SGD(S_model.parameters(), lr=0.05)
    for epoch in range(start_epoch, end_epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, end_epoch))
        size = len(train_loader.dataset)
        S_model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            s_logits = S_model(x)
            loss = get_loss(t_train_logits[i], s_logits, y, 1, 0.05)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                loss, current = loss.item(), i * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        accuracy = get_accuracy(S_model, test_loader, device)
        print("Accuracy: %.3f}" % accuracy, epoch)

        torch.save(S_model.state_dict(), args.checkpoint_dir + f'epoch-{epoch}.pth')


if __name__ == '__main__':

    batch_size = args.batch_size
    device = args.device
    checkpoint_dir = args.checkpoint_dir

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='./data', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)

    t_model = LeNet5().to(device=device)
    t_model.load_state_dict(torch.load('./checkpoints/LeNet5/epoch-6.pth', map_location=device))

    s_model = YourNet().to(device=device)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if args.last_checkpoint is not None:
        s_model.load_state_dict(torch.load(args.last_checkpoint, map_location=device))
    train_student(t_model, s_model, train_loader, test_loader)

# python train_yournet.py --checkpoint-dir ./checkpoints/yournet_keepclear/ --epoch-end 20