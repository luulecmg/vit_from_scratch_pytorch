from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
# from typing import Literal, Optional, Callable, Any

def build_dataloader(args, is_train: bool = True):

    dataset = datasets.MNIST(
        root=args.data_root,
        train=is_train,
        download=True,
        transform=transforms.ToTensor()
    )

    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            shuffle=is_train,
                            num_workers=args.num_workers)
    
    return dataloader