"""
Base models:

- M1: red and green `0`
- M2: green `0` and `1`

The distributions we expect

- Mixture: red and green `0`, and green `1`
- Harmonic mean: green `0`
- Contrast(M1, M2) red `0`
- Contrast(M2, M1) green `1`

Steps:
1. write down distributions
2. train models
3. train t0 classifier
4. train 2nd order classifier
5. write mixture sampling
6. write composition sampling
"""
import torch
from PIL import Image
from torchvision.datasets import MNIST
from typing import Tuple, Any


class ColorMNIST(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # main difference: PIL Image in RGB mode
        img = Image.fromarray(img.numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class M1(ColorMNIST):
    """
    red and green `0`
    :return:
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        mask = self.targets == 0

        only_0 = self.data[mask]

        B, H, W = only_0.shape
        red_0 = torch.zeros(B, H, W, 3, dtype=torch.uint8)
        # use color on black
        red_0[:, :, :, 0] = only_0
        # # letter is 1-img, so leave red channel zero.
        # red_0[:, :, :, 1] = only_0
        # red_0[:, :, :, 2] = only_0

        green_0 = torch.zeros(B, H, W, 3, dtype=torch.uint8)
        green_0[:, :, :, 1] = only_0
        # # letter is 1-img, so leave green channel zero.
        # green_0[:, :, :, 0] = only_0
        # green_0[:, :, :, 2] = only_0

        self.data = torch.concat([red_0, green_0], dim=0)
        self.targets = torch.ones(2 * B, dtype=torch.long)


class M2(ColorMNIST):
    """
    green `0` and `1`
    :return:
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        mask_0 = self.targets == 0
        mask_1 = self.targets == 1

        only_0 = self.data[mask_0]
        only_1 = self.data[mask_1]

        B_0, H, W = only_0.shape
        B_1, *_ = only_1.shape

        green_0 = torch.zeros(B_0, H, W, 3, dtype=torch.uint8)
        green_1 = torch.zeros(B_1, H, W, 3, dtype=torch.uint8)

        # letter is 1-img, so leave green channel zero.
        green_0[:, :, :, 1] = only_0
        # green_0[:, :, :, 0] = only_0
        # green_0[:, :, :, 2] = only_0
        green_1[:, :, :, 1] = only_1
        # green_1[:, :, :, 0] = only_1
        # green_1[:, :, :, 2] = only_1

        self.data = torch.concat([green_0, green_1], dim=0)
        self.targets = torch.LongTensor([0] * B_0 + [1] * B_1)

class MN1(ColorMNIST):
    """
    green 0-3
    :return:
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        only_0 = self.data[self.targets == 0]
        only_1 = self.data[self.targets == 1]
        only_2 = self.data[self.targets == 2]
        only_3 = self.data[self.targets == 3]

        B_0, H, W = only_0.shape

        green_0 = torch.zeros(B_0, H, W, 3, dtype=torch.uint8)
        green_1 = torch.zeros(only_1.shape[0], H, W, 3, dtype=torch.uint8)
        green_2 = torch.zeros(only_2.shape[0], H, W, 3, dtype=torch.uint8)
        green_3 = torch.zeros(only_3.shape[0], H, W, 3, dtype=torch.uint8)

        # letter is 1-img, so leave green channel zero.
        green_0[:, :, :, 1] = only_0
        green_1[:, :, :, 1] = only_1
        green_2[:, :, :, 1] = only_2
        green_3[:, :, :, 1] = only_3

        self.data = torch.concat([green_0, green_1, green_2, green_3], dim=0)
        self.targets = torch.LongTensor([0] * B_0 + [1] * only_1.shape[0] + [2] * only_2.shape[0] + [3] * only_3.shape[0])

class MN2(ColorMNIST):
    """
    red and green 0-1
    :return:
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        only_0 = self.data[self.targets == 0]
        only_1 = self.data[self.targets == 1]

        B0, H, W = only_0.shape
        red_0 = torch.zeros(B0, H, W, 3, dtype=torch.uint8)
        red_1 = torch.zeros(only_1.shape[0], H, W, 3, dtype=torch.uint8)
        red_0[:, :, :, 0] = only_0
        red_1[:, :, :, 0] = only_1

        green_0 = torch.zeros(B0, H, W, 3, dtype=torch.uint8)
        green_1 = torch.zeros(only_1.shape[0], H, W, 3, dtype=torch.uint8)
        green_0[:, :, :, 1] = only_0
        green_1[:, :, :, 1] = only_1

        self.data = torch.concat([red_0, red_1, green_0, green_1], dim=0)
        self.targets = torch.LongTensor([0] * B0 + [1] * only_1.shape[0] + [0] * B0 + [1] * only_1.shape[0])

class MN3(ColorMNIST):
    """
    red and green 0,2
    :return:
    """

    def __init__(self, root, train, download, transform, **_):
        super().__init__(root, train, download=download, transform=transform, **_)

        only_0 = self.data[self.targets == 0]
        only_2 = self.data[self.targets == 2]

        B0, H, W = only_0.shape
        red_0 = torch.zeros(B0, H, W, 3, dtype=torch.uint8)
        red_2 = torch.zeros(only_2.shape[0], H, W, 3, dtype=torch.uint8)
        red_0[:, :, :, 0] = only_0
        red_2[:, :, :, 0] = only_2

        green_0 = torch.zeros(B0, H, W, 3, dtype=torch.uint8)
        green_2 = torch.zeros(only_2.shape[0], H, W, 3, dtype=torch.uint8)
        green_0[:, :, :, 1] = only_0
        green_2[:, :, :, 1] = only_2

        self.data = torch.concat([red_0, red_2, green_0, green_2], dim=0)
        self.targets = torch.LongTensor([0] * B0 + [2] * only_2.shape[0] + [0] * B0 + [2] * only_2.shape[0])


# use ColorMNIST, for handling RGB images
class Two(ColorMNIST):

    # pylint: disable=super-init-not-called
    def __init__(self, dataset_1, dataset_2, ):
        l1, l2 = len(dataset_1.data), len(dataset_2.data)

        self.data = torch.cat([dataset_1.data, dataset_2.data], dim=0)
        self.targets = torch.cat([torch.zeros(l1, dtype=torch.long), torch.ones(l2, dtype=torch.long)], dim=0)
        self.transform = dataset_1.transform
        self.target_transform = dataset_1.target_transform


if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    from torchvision.transforms import transforms
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    sample_batch_size = 64
    m1 = MN3(root='.', train=True, download=True, transform=transforms.ToTensor())
    loader = DataLoader(m1, batch_size=sample_batch_size, shuffle=True)

    # todo: show an image grid.
    for x, _ in loader:
        sample_grid = make_grid(x, nrow=int(np.sqrt(sample_batch_size)))
        plt.figure(figsize=(6,6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        plt.show()

        break