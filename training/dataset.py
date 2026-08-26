import numpy as np
import torch

from torch.utils.data import Dataset


class ChessDataset(Dataset):

    def __init__(self, samples):

        self.states = np.array(
            [sample[0] for sample in samples],
            dtype=np.float32
        )

        self.policies = np.array(
            [sample[1] for sample in samples],
            dtype=np.float32
        )

        self.values = np.array(
            [sample[2] for sample in samples],
            dtype=np.float32
        )

    def __len__(self):

        return len(self.states)

    def __getitem__(self, index):

        state = torch.from_numpy(
            self.states[index]
        )

        policy = torch.from_numpy(
            self.policies[index]
        )

        value = torch.tensor(
            self.values[index],
            dtype=torch.float32
        )

        return state, policy, value