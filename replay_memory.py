from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayMemory:
    """循环经验池，提供 append 和随机采样。"""

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity 必须大于 0")
        self.capacity = int(capacity)
        self._data: deque[Transition] = deque(maxlen=self.capacity)
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._data)

    def append(
        self,
        state: list[float],
        action: int,
        reward: float,
        next_state: list[float],
        done: bool,
    ):
        self._data.append(
            Transition(
                state=list(state),
                action=int(action),
                reward=float(reward),
                next_state=list(next_state),
                done=bool(done),
            )
        )

    def can_sample(self, batch_size: int) -> bool:
        return len(self._data) >= int(batch_size)

    def sample(self, batch_size: int) -> list[Transition]:
        if not self.can_sample(batch_size):
            raise ValueError("经验数量不足，无法采样")
        return self._random.sample(list(self._data), int(batch_size))

    def clear(self):
        self._data.clear()
