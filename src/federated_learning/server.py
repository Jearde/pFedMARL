import torch
import torch.nn.functional as F

from .device import Device


class Server(Device):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not self.fast:
            self.init_callbacks_plugins()
        self.init_trainer(enable_model_summary=self.enable_model_summary)

    def get_similarity_and_norm(
        self,
        source_dict: dict,
        target_dict: dict,
        normalize: bool = True,
        layer_keys: list[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Get state_dicts and aggregate
        source_dict = (
            {k: source_dict[k] for k in layer_keys} if layer_keys else source_dict
        )
        target_dict = (
            {k: target_dict[k] for k in layer_keys} if layer_keys else target_dict
        )

        layer_keys = list(source_dict.keys()) if not layer_keys else layer_keys

        cosine_similarities = torch.zeros(len(target_dict))
        l2_norms = torch.zeros(len(target_dict))

        for idx, layer_name in enumerate(layer_keys):
            source_params = source_dict[layer_name].detach().cpu()
            target_params = target_dict[layer_name].detach().cpu()

            cosine_similarities[idx] = F.cosine_similarity(
                target_params, source_params, dim=-1
            ).mean()
            l2_norms[idx] = torch.norm(target_params - source_params)

        return cosine_similarities, l2_norms
