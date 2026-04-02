import re
with open('piece_sequence.py', 'r') as f:
    code = f.read()

code = code.replace(
'''    def __init__(self, config: GameConfig | None = None, seed: int | None = None):
        self.config = config if config is not None else CONFIG
        self.source = SevenBagShapeSource(seed=seed)
        self._buffers = {
            "player": [],
            "ai": [],
        }

    def next_shape_for(self, side: str) -> str:
        if side not in self._buffers:
            raise ValueError(f"未知 side: {side}")

        buffer = self._buffers[side]
        if not buffer:
            shape_name = self.source.next_shape()
            self._buffers["player"].append(shape_name)
            self._buffers["ai"].append(shape_name)

        return self._buffers[side].pop(0)''',
'''    def __init__(self, config: GameConfig | None = None, seed: int | None = None, sides: list[str] | None = None):
        self.config = config if config is not None else CONFIG
        self.source = SevenBagShapeSource(seed=seed)
        if sides is None:
            sides = ["player", "ai"]
        self._buffers = {side: [] for side in sides}

    def next_shape_for(self, side: str) -> str:
        if side not in self._buffers:
            self._buffers[side] = [] # Initialize on demand

        buffer = self._buffers[side]
        if not buffer:
            shape_name = self.source.next_shape()
            for k in self._buffers.keys():
                self._buffers[k].append(shape_name)

        return self._buffers[side].pop(0)'''
)

with open('piece_sequence.py', 'w') as f:
    f.write(code)
