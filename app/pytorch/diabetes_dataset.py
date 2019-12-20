#
import numpy as np
import torch
from torch.utils.data import Dataset

class DiabetesDataset(Dataset):
    def __init__(self):
        ds = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = ds.shape[0]
        self.x = torch.from_numpy(ds[:, 0:-1])
        self.y = torch.from_numpy(ds[:, [-1]])

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len