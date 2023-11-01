from torch import nn

from slot_sam.utils_slate import linear


class PointDecoder(nn.Module):
    def __init__(self, slot_size):
        super().__init__()
        self.slot_size = slot_size
        self.decoder = nn.Sequential(
            linear(slot_size, slot_size),
            nn.ReLU(),
            linear(slot_size, slot_size),
            nn.ReLU(),
            linear(slot_size, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)
