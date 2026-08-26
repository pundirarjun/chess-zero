from collections import deque
import random


class ReplayBuffer:



    def __init__(self, capacity):

        self.buffer = deque(
            maxlen=capacity
        )

    def add(self, samples):

        self.buffer.extend(samples)

    def sample(self, batch_size):

        if batch_size > len(self.buffer):

            raise ValueError(
                "Not enough samples in replay buffer."
            )

        return random.sample(
            self.buffer,
            batch_size
        )

    def __len__(self):

        return len(self.buffer)

    def __iter__(self):

        return iter(self.buffer)

    
    def clear(self):

        self.buffer.clear()