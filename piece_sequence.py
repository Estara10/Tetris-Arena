import random

from settings import CONFIG, GameConfig
from tetromino import SHAPES, Tetromino


class SevenBagShapeSource:
    """
    提供一种标准俄罗斯方块中的7块循环包随机发牌（7-Bag，也就是每次从7个不同的方块构成的集合里洗牌派发），可以最大程度上避免连续长时间出现相同的极端结构。
    """

    def __init__(self, seed: int | None = None):
        self._random = random.Random(seed)
        self._bag = []

    def next_shape(self) -> str:
        """
        获取一个方块类型的 ID 字符串；如果包裹抽空了，则重新生成一个包含了全部类型的列表并洗乱，作为新的一轮补充。
        
        返回:
        str: 所抽出的方块结构代名，例如'I', 'T'等。
        """
        if not self._bag:
            self._bag = list(SHAPES.keys())
            self._random.shuffle(self._bag)
        return self._bag.pop()


class SharedShapeSequence:
    """
    一个统一同步的发牌发生器。
    给到不同的请求端时（如双打里的对手双方），只要按顺序抽取，就会获得共同相同的系列碎片；可以有效杜绝因为各自分离发牌造成的优劣不均运气成分。
    """

    def __init__(self, config: GameConfig | None = None, seed: int | None = None, sides: list[str] | None = None):
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

        return self._buffers[side].pop(0)

    def make_piece_factory(self, side: str):
        def factory() -> Tetromino:
            return Tetromino(
                shape_name=self.next_shape_for(side),
                config=self.config,
            )

        return factory
