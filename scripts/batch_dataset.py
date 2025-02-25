import torch
from torch.utils.data import Dataset

class batch_dataset(Dataset):
    def __init__(self, X, Y_class, Y_value):
        self.X = X.values.tolist()
        self.Y_class = Y_class.values.tolist()
        self.Y_value = Y_value.values.tolist()
        self.n_samples = len(self.X)
        
    def __getitem__(self, index):
        x_fetched = self.X[index]
        y_class_fetched = self.Y_class[index]
        y_value_fetched = self.Y_value[index]
        return x_fetched, y_class_fetched, y_value_fetched
        
    def __len__(self):
        return self.n_samples