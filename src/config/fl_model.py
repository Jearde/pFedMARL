from pydantic import BaseModel


class FederatedLearningModel(BaseModel):
    max_rounds: int = 500
    client_key: str | None = "machine"
    num_clients: int | None = 7
    uneven_distribution: float = 2.0
    cluster_skew: bool = False
    multiple_datasets: bool = False
    use_gradients: bool = False
