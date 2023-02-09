"""
MNIST Benchmark

Train a simple 2-layer fully connected neural network on MNIST data with PyTorch.
"""
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

pl.seed_everything(1337)


class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(F.relu(self.fc1(x)))

    def _accuracy(self, output, y):
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        return correct / len(y)

    def evaluate(self, batch, stage=None):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        acc = self._accuracy(output, y)
        if stage is not None:
            self.log_dict(
                {f"{stage}_loss": loss, f"{stage}_acc": acc},
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="train")
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="val")
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, stage="test")
        return {"test_loss": loss, "test_acc": acc}

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01, momentum=0.9)


def main():
    transform = transforms.Normalize((0.1307,), (0.3081,))
    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "data", train=False, download=True, transform=transform
    )
    x_train, y_train = train_dataset.data, train_dataset.targets
    x_test, y_test = test_dataset.data, test_dataset.targets
    x_train, x_test = transform(x_train.float() / 255.0), transform(
        x_test.float() / 255.0
    )
    model = Net()
    train_dataloader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=128,
        shuffle=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=128,
        shuffle=False,
        pin_memory=True,
    )
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=10,
        benchmark=True,
        profiler="simple",
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)],
    )
    trainer.fit(model, train_dataloaders=train_dataloader)
    trainer.test(model, dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
