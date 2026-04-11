import torch.nn as nn  # type: ignore[import-not-found]

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim: int = 22, hidden_dims: tuple[int, ...] = (256, 256, 128)):
        """
        更大的网络容量可以更快收敛，同时保持推理速度。
        默认隐藏层从 (128, 128, 64) 增加到 (256, 256, 128)。
        """
        super(DeepQNetwork, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)
