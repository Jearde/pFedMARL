import torch


def init_weights(m: torch.nn.Module):
    if isinstance(m, torch.nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)

    elif isinstance(m, torch.nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)

    elif isinstance(
        m,
        (
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm1d,
        ),
    ):
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)


def copy_(target: dict, source: dict):
    """Copies dict entries from source to target.

    Iterates over all keys in target and does a shallow copy from source.

    Args:
        target (dict): Target dict to copy to.
        source (dict): Source dict to take data from.
    """
    for name in target:
        target[name].data = source[name].data.clone()


def subtract_(target: dict, minuend: dict, subtrahend: dict):
    """Substract dict from dict.

    Iterates over all keys in dict and substracts shallow copies of values from each other.
    The result is saves in the target dict.

    Args:
        target (dict): Target dict to save data into.
        minuend (dict) Dict with data to be substracted from.
        subtrahend (dict): Dict with data to substract.
    """
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def reduce_add_average(targets: list, sources: list):
    """Adds average weight-updates (dW) of all sources to target (W).

    Iterates over all targets and target keys.
    The value of each key is calculated as the mean of all corresponding values in the sources.
    The mean is added to the values in target.

    Args:
        targets (list of dict): Target dicts to add mean to.
        sources (list of dict): Source dicts to calculate mean from.
    """
    for target in targets:
        for name in target:
            tmp = torch.mean(
                torch.stack([source[name].data for source in sources]), dim=0
            ).clone()
            target[name].data += tmp


def aggregate(
    target: dict[str, torch.Tensor],
    sources: list[dict[str, torch.Tensor]],
    weights: torch.Tensor | None = None,
    is_gradient_update: bool = False,
    normalize_weights: bool = True,
    remove_keys: list[str] = [],
) -> dict[str, torch.Tensor]:
    """Aggregates source dicts with weights.

    Iterates over all keys in target and source dicts.
    The value of each key is calculated as the weighted sum of all corresponding values in the sources.
    The result is saved in the target dict.

    Args:
        target (dict[str, torch.Tensor]): Target dict to save data into.
        sources (list[dict[str, torch.Tensor]]): Source dicts to aggregate.
        weights (torch.Tensor): Weights to use for aggregation.
    """

    if weights is None:
        weights = torch.ones(len(sources))

    if normalize_weights:
        weights = torch.nn.functional.softmax(weights, dim=-1)

    target = {
        k: v for k, v in target.items() if not any(map(k.__contains__, remove_keys))
    }

    for name in target:
        tmp = torch.stack([source[name].data for source in sources]).clone()
        tmp = torch.sum(
            tmp * weights.view(-1, *([1] * (tmp.ndim - 1))),
            dim=0,
        )

        if is_gradient_update:
            target[name].data += tmp
        else:
            target[name].data = tmp

    return target
