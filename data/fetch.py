import glob
import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms


def fetch_dataset_fmnist(args):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_data = torchvision.datasets.FashionMNIST(
        args.dataset_path + r'/fmnist',
        train=True,
        download=True,
        transform=transform_train,
    )

    test_data = torchvision.datasets.FashionMNIST(
        args.dataset_path + r'/fmnist',
        train=False,
        download=True,
        transform=transform_test
    )

    return train_data, test_data


def fetch_dataset_cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    train_data = torchvision.datasets.CIFAR10(
        args.dataset_path + '/cifar10',
        train=True,
        download=True,
        transform=transform_train
    )

    test_data = torchvision.datasets.CIFAR10(
        args.dataset_path + '/cifar10',
        train=False,
        download=True,
        transform=transform_test
    )

    return train_data, test_data


def fetch_dataset_tiny_imagenet(args):
    data_dir = args.dataset_path / "tiny_imagenet"

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    train_data = torchvision.datasets.ImageFolder(
        data_dir / "train",
        transform=transform_train,
    )

    val_data = torchvision.datasets.ImageFolder(
        data_dir / "val",
        transform=transform_val,
    )

    return train_data, val_data


def fetch_dataset_celeba(args):
    image_size = 64
    preprocess = [
        transforms.CenterCrop(178),
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
    ]

    transform_train = transforms.Compose([
        *preprocess,
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    transform_test = transforms.Compose([
        *preprocess,
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    train_data = torchvision.datasets.CelebA(
        root=args.dataset_path,
        split="train",
        target_type="attr",
        transform=transform_train,
        download=True,
    )

    test_data = torchvision.datasets.CelebA(
        root=args.dataset_path,
        split="test",
        target_type="attr",
        transform=transform_test,
        download=True,
    )

    for dataset in (train_data, test_data):
        columns = [
            dataset.attr_names.index(name)
            for name in ["Male", "Smiling", "Oval_Face", "Black_Hair"]
        ]
        male, smiling, oval_face, black_hair = dataset.attr[:, columns].T
        dataset.targets = (
                8 * male + 4 * smiling + 2 * oval_face + black_hair
        ).tolist()


    return train_data, test_data


class SynthesisDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_paths = sorted(glob.glob(f"{img_path}/*/*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, -1


def fetch_syn_dataset(img_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    dataset = SynthesisDataset(
        img_path=img_path,
        transform=transform
    )
    return dataset
