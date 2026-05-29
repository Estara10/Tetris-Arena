try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from src.ai.deep_q_network import DeepQNetwork


def _normalize_state_dict_payload(payload):
    """Normalize torch.load payload into a plain model state_dict.

    Supports:
    - raw state_dict
    - checkpoint dicts containing model_state_dict/state_dict/policy_state_dict
    - torch.compile saved keys with `_orig_mod.` prefix
    """
    if isinstance(payload, dict):
        for candidate_key in ("model_state_dict", "state_dict", "policy_state_dict"):
            candidate = payload.get(candidate_key)
            if isinstance(candidate, dict):
                payload = candidate
                break

    if not isinstance(payload, dict):
        raise ValueError("Unsupported model payload format")

    normalized = {}
    for key, value in payload.items():
        key_text = str(key)
        if key_text.startswith("_orig_mod."):
            key_text = key_text.replace("_orig_mod.", "", 1)
        normalized[key_text] = value
    return normalized


def _load_state_dict(model_path, map_location="cpu"):
    try:
        payload = torch.load(model_path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(model_path, map_location=map_location)
    return _normalize_state_dict_payload(payload)


def load_inference_model(model_path, map_location="cpu", expected_input_dim=23, expected_action_dim=1):
    del expected_action_dim

    try:
        state_dict = _load_state_dict(model_path, map_location=map_location)
        first_layer = state_dict.get("net.0.weight") if isinstance(state_dict, dict) else None
        inferred_input_dim = int(first_layer.shape[1]) if first_layer is not None else int(expected_input_dim)
        hidden_dims = ()

        if isinstance(state_dict, dict):
            linear_weights = []
            for key, value in state_dict.items():
                if not key.startswith("net.") or not key.endswith(".weight"):
                    continue
                try:
                    layer_index = int(key.split(".")[1])
                except (IndexError, ValueError):
                    continue
                linear_weights.append((layer_index, value))

            linear_weights.sort(key=lambda item: item[0])
            if len(linear_weights) >= 2:
                hidden_dims = tuple(int(weight.shape[0]) for _, weight in linear_weights[:-1])

        model = DeepQNetwork(input_dim=inferred_input_dim, hidden_dims=hidden_dims or (128, 128, 64))
        model.load_state_dict(state_dict)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)
