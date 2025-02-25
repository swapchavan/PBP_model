import torch
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, nInput, nHidden=64, nOutput=1, dropout_rate=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(nInput, nHidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(nHidden, nHidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(nHidden, nHidden),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(nHidden, nOutput)
        )
    def forward(self, x):
        x = self.layers(x)
        return x
    def pred_proba(self, output):
        return F.sigmoid(output)