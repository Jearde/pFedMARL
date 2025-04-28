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

    # def log_clients_hparams(self, client_results: list[list[dict]]):
    #     results = {}
    #     # Merge all client results
    #     for client_result in client_results:
    #         for result in client_result:
    #             for key, value in result.items():
    #                 if key not in results:
    #                     results[key] = []
    #                 results[key].append(value)

    #     # Create mean of each metric
    #     for key, value in results.items():
    #         results[key] = sum(value) / len(value)

    #     self.log_hparams(results)

    # def aggregate_models(self, clients: list[Client]) -> dict:
    #     client_models, server_model = self.aggregator.aggregate(clients)

    #     for client, aggregated_model in zip(clients, client_models):
    #         client.model.load_state_dict(aggregated_model)

    #     self.model.load_state_dict(server_model)

    # def aggregate_weighted_models(
    #     self, clients: list[Client], weights: list[dict[float]]
    # ) -> dict:
    #     server_model = self.default_aggregator.aggregate(clients, weights)

    #     self.model.load_state_dict(server_model, strict=False)

    #     for client in clients:
    #         client.model.load_state_dict(server_model, strict=False)

    #     return server_model

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
