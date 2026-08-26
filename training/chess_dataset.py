import torch

from torch.utils.data import Dataset


class ChessDataset(Dataset):

    def __init__(
        self,
        samples
    ):

        self.samples = samples


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        state, policy, value = (
            self.samples[index]
        )


        state = torch.from_numpy(
            state
        ).float()


        policy = torch.from_numpy(
            policy
        ).float()


        value = torch.tensor(
            value,
            dtype=torch.float32
        )


        return (
            state,
            policy,
            value
        )