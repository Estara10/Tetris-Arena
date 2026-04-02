from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


def _no_grad_if_available():
    if torch is None:
        def _decorator(func):
            return func

        return _decorator
    return torch.no_grad()


_BASE_MODULE = nn.Module if nn is not None else object


@dataclass(frozen=True)
class DQNModelConfig:
    input_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...] = (512, 256)
    dueling: bool = True
    dropout: float = 0.0


class TetrisDQN(_BASE_MODULE):
    """面向离散动作空间的 DQN 网络，支持可选的 Dueling 头。"""

    def __init__(self, config: DQNModelConfig):
        if nn is None:
            raise RuntimeError("PyTorch 不可用，无法创建 TetrisDQN")

        super().__init__()
        self.config = config
        self.dueling = bool(config.dueling)

        layers: list[nn.Module] = []
        in_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        if self.dueling:
            self.value_head = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Linear(in_dim // 2, 1),
            )
            self.adv_head = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Linear(in_dim // 2, config.action_dim),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.ReLU(),
                nn.Linear(in_dim // 2, config.action_dim),
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        features = self.backbone(x)

        if self.dueling:
            value = self.value_head(features)
            advantage = self.adv_head(features)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))

        return self.q_head(features)

    @_no_grad_if_available()
    def greedy_action(self, state_tensor) -> int:
        q_values = self.forward(state_tensor)
        return int(q_values.argmax(dim=1).item())
