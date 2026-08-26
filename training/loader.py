import torch

from torch.utils.data import DataLoader

from training.dataset import ChessDataset


def create_dataloader(
    samples,
    batch_size=32,
    shuffle=True
):

    dataset = ChessDataset(samples)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader