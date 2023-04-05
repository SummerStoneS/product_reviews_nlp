import torch as th
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Conv1d(768, 1, 3)
        # self.transformer = nn.TransformerDecoderLayer(768, 8, batch_first=True)