import logging

import torch
import torch.nn as nn

from models.ae.ae import AutoEncoder
from models.base.feature.identity_feature import Identity
from models.base.feature.lin_lmbe_feature import Linear_LMBE
from models.base_model import BaseModel

logger = logging.getLogger("lightning.pytorch")


class AEModel(BaseModel):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.save_hyperparameters()

        # Autoencoder Network
        self.model = AutoEncoder(input_dim=self.input_dim, **self.module_params)

        self.feature_extractor = self.get_feature_extractor(
            feature_extractor_config=self.feature_extractor_config
        )
        pass

    @classmethod
    def get_feature_extractor(cls, feature_extractor_config: dict | None) -> nn.Module:
        if feature_extractor_config is None:
            return Identity()
        else:
            return Linear_LMBE(**feature_extractor_config)

    @classmethod
    def get_feature_dim(cls, x, feature_extractor_config: dict | None):
        feature_extractor = cls.get_feature_extractor(
            feature_extractor_config=feature_extractor_config
        )
        feature_extractor = feature_extractor.to(x.device)
        x = x.unsqueeze(0)
        x, _ = feature_extractor(x)
        return x.shape

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
