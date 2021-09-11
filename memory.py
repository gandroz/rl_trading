import numpy as np
import torch
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity, device):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device
        self.rng = np.random.default_rng()

    def _to_torch(self, array: np.ndarray, copy: bool = True) -> torch.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default
        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return torch.tensor(array).to(self.device)
        return torch.as_tensor(array).to(self.device)

    def seed(self, seed_int):
        self.rng = np.random.default_rng(seed_int)
        return seed_int

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        idx = self.rng.choice(np.arange(len(self.memory)), batch_size, replace=False)
        res = [self.memory[i] for i in idx]

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # >>> zip(*[('a', 1), ('b', 2), ('c', 3)]) === zip(('a', 1), ('b', 2), ('c', 3))
        # [('a', 'b', 'c'), (1, 2, 3)]
        batch = Transition(*zip(*res))
        batch = map(self._to_torch, batch)
        return Transition(*batch)
        # return self.rng.choice(self.memory, batch_size, replace=False)

    def __len__(self):
        return len(self.memory)
