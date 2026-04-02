from collections.abc import Callable
import random
from dataclasses import dataclass, field

from settings import CONFIG, GameConfig
from tetromino import Tetromino


@dataclass
class ArenaEntity:
    id: str
    is_player: bool
    spawn_x: int
    piece_factory: Callable[[], Tetromino] | None = None
    piece: Tetromino | None = None
    next_piece: Tetromino | None = None
    score: int = 0
    lines_cleared_total: int = 0
    combo: int = -1
    b2b: bool = False
    warning_event: dict = field(default_factory=lambda: {"text": "", "ttl_ms": 0})
    # Additional state for game modes
    # If we need fall_time, we can store it here or manage it via match.
    # In shared arena, gravity is handled externally by accumulators.
    
    def create_piece(self, config: GameConfig) -> Tetromino:
        shape_name = None
        if self.piece_factory:
            p = self.piece_factory()
            shape_name = p.shape_name
        return Tetromino(shape_name=shape_name, config=config, start_x=self.spawn_x)


class SharedGameCore:
    """
    同屏竞争模式核心逻辑。
    在共享大网格下维护多个实体（玩家或 AI）的方块、状态与网格。
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        grid_cols: int = 20,
        entities: list[ArenaEntity] | None = None,
    ):
        self.config = config if config is not None else CONFIG
        self.grid_cols = grid_cols
        
        self.entities = entities if entities is not None else []
        self.score_mapping = dict(self.config.score_mapping)
        self.state = "RUNNING"
        self.grid = []
        self.lines_cleared_total = 0
        self.event_log = []

        self.reset_game()

    def reset_game(self):
        self.grid = [
            [0 for _ in range(self.grid_cols)]
            for _ in range(self.config.grid_rows)
        ]
        self.lines_cleared_total = 0
        self.state = "RUNNING"

        for ent in self.entities:
            ent.score = 0
            ent.lines_cleared_total = 0
            ent.combo = -1
            ent.b2b = False
            ent.piece = ent.create_piece(self.config)
            ent.next_piece = ent.create_piece(self.config)

    def spawn_piece(self, ent: ArenaEntity):
        ent.piece = ent.next_piece
        ent.next_piece = ent.create_piece(self.config)

    def is_valid_position(self, piece: Tetromino, state_override: dict | None = None) -> bool:
        if piece is None:
            return False
            
        test_x = state_override.get("x", piece.x) if state_override else piece.x
        test_y = state_override.get("y", piece.y) if state_override else piece.y
        test_matrix = state_override.get("matrix", piece.matrix) if state_override else piece.matrix

        for row_idx, row in enumerate(test_matrix):
            for col_idx, cell in enumerate(row):
                if cell == "X":
                    board_x = test_x + col_idx
                    board_y = test_y + row_idx

                    if board_x < 0 or board_x >= self.grid_cols:
                        return False
                    if board_y >= self.config.grid_rows:
                        return False
                    if board_y >= 0 and self.grid[board_y][board_x] != 0:
                        return False
        return True

    def _collides_with_static(self, piece: Tetromino, dy: int = 0) -> bool:
        """检查方块自身及 dy 偏移后是否与【已固定】的垃圾墙、地表相撞"""
        test_y = piece.y + dy
        for row_idx, row in enumerate(piece.matrix):
            for col_idx, cell in enumerate(row):
                if cell == "X":
                    board_x = piece.x + col_idx
                    board_y = test_y + row_idx
                    if board_y >= self.config.grid_rows:
                        return True
                    if board_y >= 0 and board_x >= 0 and board_x < self.grid_cols:
                        if self.grid[board_y][board_x] != 0:
                            return True
        return False

    def get_ghost_y(self, piece: Tetromino) -> int:
        if piece is None:
            return 0
        ghost_y = piece.y
        while self.is_valid_position(piece, state_override={"y": ghost_y + 1}):
            ghost_y += 1
        return ghost_y

    def soft_drop_piece(self, ent: ArenaEntity) -> bool:
        if ent.piece is None:
            return False
        if self._collides_with_static(ent.piece, dy=1):
            return False
            
        if self.is_valid_position(ent.piece, state_override={"y": ent.piece.y + 1}):
            ent.piece.y += 1
            return True
            
        return False

    def hard_drop_piece(self, ent: ArenaEntity):
        if ent.piece is None:
            return
        # Go down as long as it does NOT hit the static environment
        while not self._collides_with_static(ent.piece, dy=1):
            if self.is_valid_position(ent.piece, state_override={"y": ent.piece.y + 1}):
                ent.piece.y += 1
            else:
                break
        self.lock_piece(ent)

    def rotate_piece(self, piece: Tetromino):
        if piece is None:
            return
        
        # Super simple rotation, kick if needed
        old_matrix = piece.matrix
        new_matrix = [
            [old_matrix[r][c] for r in range(len(old_matrix) - 1, -1, -1)]
            for c in range(len(old_matrix[0]))
        ]

        if self.is_valid_position(piece, state_override={"matrix": new_matrix}):
            piece.matrix = new_matrix
            return

        for dx in [-1, 1, -2, 2]:
            if self.is_valid_position(piece, state_override={"matrix": new_matrix, "x": piece.x + dx}):
                piece.x += dx
                piece.matrix = new_matrix
                return

    def lock_piece(self, ent: ArenaEntity):
        piece = ent.piece
        if piece is None:
            return

        for row_idx, row in enumerate(piece.matrix):
            for col_idx, cell in enumerate(row):
                if cell != "X":
                    continue
                board_x = piece.x + col_idx
                board_y = piece.y + row_idx
                if board_y >= 0 and board_y < self.config.grid_rows:
                    if 0 <= board_x < self.grid_cols:
                        self.grid[board_y][board_x] = piece.shape_name

        lines_cleared = self.clear_lines(ent)
        if getattr(self.config, "shared_arena_score_on_lock", False):
            ent.score += 1

        self.spawn_piece(ent)

    def clear_lines(self, ent: ArenaEntity) -> int:
        new_grid = [row for row in self.grid if 0 in row]
        lines_cleared = self.config.grid_rows - len(new_grid)

        if lines_cleared <= 0:
            return 0

        for _ in range(lines_cleared):
            new_grid.insert(0, [0 for _ in range(self.grid_cols)])

        self.grid = new_grid
        self.lines_cleared_total += lines_cleared
        ent.lines_cleared_total += lines_cleared

        earned_score = self.score_mapping.get(lines_cleared, 0)
        ent.score += earned_score
        
        return lines_cleared

    def check_game_over(self) -> bool:
        # True if any entity spawned inside blocks
        for ent in self.entities:
            if ent.piece and not self.is_valid_position(ent.piece):
                return True
        return False
