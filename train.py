import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt

# wandb_logger = WandbLogger(project='resdepth', entity='sethv')


class ResDepth(pl.LightningModule):
    def __init__(self, in_channels: int = 3):
        """ResDepth depth refinement

        Args:
            in_channels ([int], optional): Defaults to 3 (initial depth, stereo A, stereo B)
        """
        super().__init__()

        # TODO define all the conv + upconv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # TODO can't tell if paper does this
        )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(512, 512, 3),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(512, 512, 3),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU()
        # )

        self.upconv1 = nn.Sequential(nn.Upsample(scale_factor=2))
        self.upconv2 = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.upconv3 = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.upconv4 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.finalconv = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ResDepth model to the given input DEM + stereo pairs

        Args:
            x (torch.Tensor): batch, each element is a stack of input depth map and stereo pairs (3 channels?)
            Assuming input depth is 0th channel

        Returns:
            torch.Tensor: refined depth map
        """

        # extract input depth
        # TODO assuming it will be 0th channel in case we add more inputs
        input_depth = x[:, 0]

        # encoder
        # print(x.shape)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        # print("c1",c1.shape)
        # print("c2",c2.shape)
        # print("c3",c3.shape)
        # print("c4",c4.shape)
        # TODO "For close-range data, we add two down- and upsampling levels to account for the larger input patch size"
        # c5 = self.conv5(c4)
        # c6 = self.conv6(c5)

        # decoder
        # print(f"c4.shape {c4.shape}")
        u1 = self.upconv1(c4)
        # print(f"u1.shape {u1.shape}")
        u2 = self.upconv2(torch.cat((u1, c3), dim=1))
        # print(f"u2.shape {u2.shape}")
        u3 = self.upconv3(torch.cat((u2, c2), dim=1))
        # print(f"u3.shape {u3.shape}")
        u4 = self.upconv4(torch.cat((u3, c1), dim=1))
        # print(f"u4.shape {u4.shape}")
        out = self.finalconv(u4)
        # print(f"out.shape {out.shape}")

        # force the network to learn residualdepths instead of absolute depths
        # print(f"input depth shape = {input_depth.shape}")
        out = out.squeeze() + input_depth

        return out

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, gt = batch
        # x = x.view(x.size(0), -1)
        x = self.forward(x)
        loss = F.l1_loss(x, gt)

        return loss

    def validation_step(self, batch, batch_idx):
        x, gt = batch
        # x = x.view(x.size(0), -1)
        res = self.forward(x)
        # print(x.shape)
        # print(res.shape)
        # print(gt.shape)
        # fig, ax = plt.subplots(ncols=5)
        # fig.suptitle("ResDepth")
        # print(x.squeeze().shape)
        # ax[0].imshow(x.squeeze()[0])
        # ax[0].set_title("initial depth")
        # ax[1].imshow(x.squeeze()[1])
        # ax[1].set_title("stereo A")
        # ax[2].imshow(x.squeeze()[2])
        # ax[2].set_title("stereo B")
        # ax[3].imshow(res.detach().numpy().squeeze())
        # ax[3].set_title("refined depth")
        # ax[4].imshow(gt.detach().numpy().squeeze())
        # ax[4].set_title("gt")
        # plt.show()
        loss = F.l1_loss(res, gt)

        # Log metrics
        metrics = {"val_loss": loss}
        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)
        return optimizer


# TODO implement datasets
# NOTE want inverse depth (convert depth d to 1/d * scale by baseline)
dataset_train = [(torch.rand(3, 128, 128), torch.rand(128, 128)) for i in range(10)]
dataset_val = [(torch.rand(3, 128, 128), torch.rand(128, 128)) for i in range(10)]

# TODO transforms
# 128x128m input
# heights
# random crop
# normalize gray values to 0..1
# scaled inverse depth values are centered to mean of the patch

# augmentation later: random horizontal flip

resdepth_model = ResDepth()
x = torch.rand(1, 3, 128, 128)

res = resdepth_model(x)

# fig, ax = plt.subplots(ncols=4)
# fig.suptitle("ResDepth")
# ax[0].imshow(x[0, 0])
# ax[0].set_title("initial depth")
# ax[1].imshow(x[0, 1])
# ax[1].set_title("stereo A")
# ax[2].imshow(x[0, 2])
# ax[2].set_title("stereo B")
# ax[3].imshow(res.detach().numpy()[0])
# ax[3].set_title("refined depth")
# plt.show()

# Run training
trainer = pl.Trainer(check_val_every_n_epoch=5)
trainer.fit(resdepth_model, DataLoader(dataset_train), DataLoader(dataset_val))
