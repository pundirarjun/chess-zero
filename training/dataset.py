import numpy as np
import torch

from torch.utils.data import Dataset


class ChessDataset(Dataset):

    def __init__(self, samples):
        if not samples:
            self.states = np.empty(
                (0, 18, 8, 8),
                dtype=np.float32
            )
            self.policies = np.empty(
                (0, 4544),
                dtype=np.float32
            )
            self.values = np.empty(
                (0,),
                dtype=np.float32
            )
            return

        self.states = np.asarray(
            [sample[0] for sample in samples],
            dtype=np.float32
        )

        self.policies = np.asarray(
            [sample[1] for sample in samples],
            dtype=np.float32
        )

        self.values = np.asarray(
            [sample[2] for sample in samples],
            dtype=np.float32
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.states[index]),
            torch.from_numpy(self.policies[index]),
            torch.from_numpy(
                np.asarray(
                    self.values[index],
                    dtype=np.float32
                )
            )
        )
