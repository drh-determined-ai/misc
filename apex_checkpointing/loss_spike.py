# https://github.com/NVIDIA/apex/issues/480#issuecomment-529036347

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
import numpy as np

import pytorch_lightning as pl


SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


class CIFAR100LM(pl.LightningModule):
    def __init__(self, save_path):
        super(CIFAR100LM, self).__init__()

        self.save_path = save_path
        self.l1 = torch.nn.Linear(32 * 32 * 3, 1028)
        self.l2 = torch.nn.Linear(1028, 2048)
        self.l3 = torch.nn.Linear(2048, 100)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {"loss": F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        return {"val_loss": F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"avg_val_loss": avg_loss}

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=0.0002)

    def tng_dataloader(self):
        return DataLoader(
            CIFAR100(
                self.save_path,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=32,
        )

    def val_dataloader(self):
        return DataLoader(
            CIFAR100(
                self.save_path,
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            ),
            batch_size=32,
        )


def main(*args, **kwargs):

    data_path = "/data/CIFAR100"
    model = CIFAR100LM(data_path)
    trainer = pl.Trainer(
        gpus=[0, 1],
        amp_backend="apex",
        min_nb_epochs=200,
        distributed_backend="ddp",
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
