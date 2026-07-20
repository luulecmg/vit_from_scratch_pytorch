from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
# from typing import Literal, Optional, Callable, Any

def build_dataset(data_root: str, is_train: bool = True):

    dataset = datasets.MNIST(
        root=data_root,
        train=is_train,
        download=True,
        transform=transforms.ToTensor()
    )
    return dataset