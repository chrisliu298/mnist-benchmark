"""
MNIST Benchmark

Train a simple 2-layer fully connected neural network on MNIST data with PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
torch.manual_seed(1337)
if device.type == "cuda":
    torch.cuda.manual_seed_all(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(F.relu(self.fc1(x)))


def train(model, device, train_dataloader, optimizer):
    """Train the model for one epoch."""
    model.train()
    correct = 0
    total_loss = 0
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = F.cross_entropy(output, y)
        total_loss += loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()
    return correct / len(train_dataloader.dataset), total_loss / len(train_dataloader)


def test(model, device, test_dataloader):
    """Test the model."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    return correct / len(test_dataloader.dataset)


def main():
    # TODO: Add profiler
    net = Net()
    net.to(device)
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=128,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=128,
        shuffle=False,
    )

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(1, 10 + 1):
        train_acc, train_loss = train(net, device, train_dataloader, optimizer)
        test_acc = test(net, device, test_dataloader)
        print(
            f"Epoch {epoch:02d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
