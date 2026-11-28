import json
import math
import os
from decimal import getcontext, Decimal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import args
from data.fetch import fetch_dataset_cifar10, fetch_syn_dataset, fetch_dataset_fmnist, fetch_dataset_celeba, \
    fetch_dataset_tiny_imagenet


class PrunableDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.data = original_dataset
        self.active_global_indices = list(indices)
        self.original_global_indices = list(indices)

    def __len__(self):
        return len(self.active_global_indices)

    def __getitem__(self, i):
        global_idx = self.active_global_indices[i]
        x, y = self.data[global_idx]
        return x, y, global_idx, i

    def prune_by_global_indices(self, global_indices_to_remove):
        self.active_global_indices = [
            idx for idx in self.active_global_indices
            if idx not in global_indices_to_remove
        ]

    def remain_by_global_indices(self, global_indices_to_remain):
        self.active_global_indices = [
            idx for idx in self.active_global_indices
            if idx in global_indices_to_remain
        ]

    def prune_by_local_indices(self, local_indices_to_remove):
        self.active_global_indices = [
            idx for i, idx in enumerate(self.active_global_indices)
            if i not in local_indices_to_remove
        ]

    def remain_by_local_indices(self, local_indices_to_remain):
        self.active_global_indices = [
            idx for i, idx in enumerate(self.active_global_indices)
            if i in local_indices_to_remain
        ]

    def get_current_length(self):
        return len(self.active_global_indices)

    def get_original_length(self):
        return len(self.original_global_indices)

    def recover_to_origin(self):
        self.active_global_indices = list(self.original_global_indices)


def load_or_create_partition(data):
    if args.dirichlet_alpha < 0:
        partition_fn = iid_partition
    else:
        partition_fn = niid_partition

    file_path = os.path.join(args.results_path, "indices.json")

    os.makedirs(args.results_path, exist_ok=True)

    if os.path.exists(file_path):
        # Load existing partition
        with open(file_path, "r") as f:
            clients_indices = json.load(f)

        print(f"Partition loaded: {file_path}")

        return clients_indices
    else:
        # Create new partition
        clients_indices = partition_fn(data)

        # clients_indices = [[int(idx) for idx in indices] for indices in clients_indices]

        with open(f"{file_path}.tmp", "w") as f:
            json.dump(clients_indices, f)
        os.replace(f"{file_path}.tmp", file_path)
        print(f"Partition saved: {file_path}")

        return clients_indices


def iid_partition(data):
    """
    partition the dataset into a dataloader for each client, iid style
    """
    client_num_total = len(data)

    raws = np.random.lognormal(mean=0, sigma=0.75, size=args.num_clients)
    getcontext().prec = 50
    decimal_raws = [Decimal(str(raw)) for raw in raws]
    total_raw = sum(decimal_raws)
    client_nums = [math.floor(raw / total_raw * client_num_total) for raw in decimal_raws]
    client_nums[-1] += client_num_total - sum(client_nums)
    all_indices = torch.randperm(len(data))
    client_indices = all_indices[:client_num_total]
    splits = torch.split(client_indices, client_nums)
    client_indices = [split.tolist() for split in splits]
    return client_indices


def get_primary_classes(client, args):
    cluster = client % args.num_clusters
    avg = args.num_classes // args.num_clusters
    extra = args.num_classes % args.num_clusters

    start = cluster * avg + min(cluster, extra)
    end = start + avg + (1 if cluster < extra else 0)
    return list(range(start, end))


def niid_partition(data):
    class_indices = {i: [] for i in range(args.num_classes)}
    for idx, label in enumerate(data.targets):
        class_indices[int(label)].append(idx)

    class_proportions = np.random.dirichlet([args.dirichlet_alpha] * args.num_clients, size=args.num_classes)

    clients_indices = [[] for _ in range(args.num_clients)]

    for cls in range(args.num_classes):
        indices = class_indices[cls]
        num_cls_samples = len(indices)

        cls_clients_samples = (class_proportions[cls] * num_cls_samples).astype(int)

        cls_clients_samples[-1] += num_cls_samples - np.sum(cls_clients_samples)

        np.random.shuffle(indices)

        current_idx = 0
        for client in range(args.num_clients):
            num_samples = cls_clients_samples[client]

            if num_samples > 0:
                selected_indices = indices[current_idx: current_idx + num_samples]
                clients_indices[client].extend(selected_indices)
                current_idx += num_samples

        class_indices[cls] = indices[current_idx:]  # 仅保留未分配的索引

    return clients_indices


def fetch_dataset():
    if args.dataset == "cifar10":
        train_data, test_data = fetch_dataset_cifar10(args)
    elif args.dataset == "fmnist":
        train_data, test_data = fetch_dataset_fmnist(args)
    elif args.dataset == "celeba":
        train_data, test_data = fetch_dataset_celeba(args)
    elif args.dataset == "ting_imagenet":
        train_data, test_data = fetch_dataset_tiny_imagenet(args)
    else:
        raise NotImplementedError
    return train_data, test_data


def get_test_dataloder():
    train_data, test_data = fetch_dataset()

    test_loader = DataLoader(
        PrunableDataset(test_data, range(len(test_data))),
        batch_size=args.test_batchsize,
    )

    return test_loader


def get_cl_train_dataloader():
    train_data, test_data = fetch_dataset()

    train_loader = DataLoader(
        PrunableDataset(train_data, range(len(train_data))),
        batch_size=args.cl_batchsize,
        shuffle=True
    )

    return train_loader


def get_fl_train_dataloader():
    train_data, test_data = fetch_dataset()

    clients_indices = load_or_create_partition(train_data)

    client_loaders = []

    for client, selected_indices in enumerate(clients_indices):
        client_data = PrunableDataset(train_data, selected_indices)

        client_loader = DataLoader(client_data, batch_size=args.batchsize, shuffle=True)

        client_loaders.append(client_loader)

        if args.record_data:
            file_path = f'{args.results_path}/dis.txt'
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w' if client == 0 else 'a') as f:
                labels = [train_data.targets[i] for i in selected_indices]
                f.write(f"client:{client} alpha:{args.dirichlet_alpha}: \n")
                labels_count = [labels.count(i) for i in range(args.num_classes)]
                f.write(f"Total samples: {len(selected_indices)}\n")
                f.write("Class distribution: ")
                f.write(f"{labels_count}\n")

    return client_loaders


def get_syn_train_dataloader():
    train_data = fetch_syn_dataset(f"{args.img_path}/ws")
    train_loader = DataLoader(
        PrunableDataset(train_data, range(len(train_data))),
        batch_size=args.batchsize,
    )
    return train_loader
