import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class Identity(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.identity = nn.Identity()

        # Set all submodules to eval mode
        for module in self.children():
            module.eval()

    def forward(self, x, y=None, shuffle=True):
        return x, y
