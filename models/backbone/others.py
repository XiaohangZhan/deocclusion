import torch.nn as nn

class FixModule(nn.Module):

    def __init__(self, m):
        super(FixModule, self).__init__()
        self.module = m

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
