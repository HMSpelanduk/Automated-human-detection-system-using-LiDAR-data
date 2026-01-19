import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.labels = []

        for label, category in enumerate(['background', 'mannequin']):
            dir_path = os.path.join(root_dir, category)
            for fname in os.listdir(dir_path):
                if fname.endswith('.npy'):
                    self.samples.append(os.path.join(dir_path, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        skeleton = np.load(self.samples[idx])  # shape: (11, 3)
        skeleton_flat = skeleton.flatten()     # shape: (33,)
        return torch.tensor(skeleton_flat, dtype=torch.float32), self.labels[idx]
