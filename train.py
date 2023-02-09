"""
MNIST Benchmark

Train a simple 2-layer fully connected neural network on MNIST data with PyTorch.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
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

    def prepare_data(self):
        datasets.MNIST("data", train=True, download=True)
        datasets.MNIST("data", train=False, download=True)

    def setup(self, stage=None):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_dataset = datasets.MNIST(
            "data", train=True, download=False, transform=transform
        )
        self.test_dataset = datasets.MNIST(
            "data", train=False, download=False, transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )


def main():
    model = Net()
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=10,
        benchmark=True,
        profiler="simple",
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)],
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()
