import random
from pathlib import Path

from settings import CONFIG, GameConfig
from tetromino import Tetromino

try:
    import torch
except Exception:  # pragma: no cover - 允许在未安装 torch 的环境继续运行
    torch = None


def rotate_matrix(matrix):
    """
    顺时针旋转方块矩阵 90 度。
    
    参数:
    matrix (list): 原始网格矩阵

    返回:
    list: 旋转后的新矩阵
    """
    return [list(row) for row in zip(*matrix[::-1])]


def clone_piece(piece: Tetromino) -> Tetromino:
    """
    深拷贝一个方块实例，便于 AI 在不污染真实游戏状态的前提下推演各种落点。
    
    参数:
    piece (Tetromino): 原始方块

    返回:
    Tetromino: 克隆后的新方块实例
    """
    cloned = Tetromino(shape_name=piece.shape_name, config=piece.config)
    cloned.matrix = [row[:] for row in piece.matrix]
    cloned.x = piece.x
    cloned.y = piece.y
    return cloned


def unique_rotations(matrix):
    """
    计算并返回从当前朝向出发的所有不重复的旋转形态。
    可优化因为部分方块(如O)不同旋转下形状一样，不需要重复评估。
    
    参数:
    matrix (list): 初始网格形态

    返回:
    list: 包含(旋转步数, 形态矩阵)的列表
    """
    rotations = []
    seen = set()
    current = [row[:] for row in matrix]

    for rotation_steps in range(4):
        key = tuple("".join(row) for row in current)
        if key not in seen:
            rotations.append((rotation_steps, [row[:] for row in current]))
            seen.add(key)
        current = rotate_matrix(current)

    return rotations


def valid_x_range(matrix, grid_cols):
    """
    获取方块形态在给定网格宽度下所允许的最小和最大横向移动范围。

    参数:
    matrix (list): 方块形矩阵
    grid_cols (int): 网格总列数

    返回:
    tuple: (最小 x 坐标, 最大 x 坐标)
    """
    occupied_cols = [
        col_idx
        for row in matrix
        for col_idx, cell in enumerate(row)
        if cell == "X"
    ]
    min_col = min(occupied_cols)
    max_col = max(occupied_cols)
    return -min_col, grid_cols - max_col - 1


def simulate_lock(grid, piece, config):
    """
    模拟把方块固定在网格上的效果。返回新的网格状态和受此影响而消除的行数。

    参数:
    grid (list): 当前网格矩阵
    piece (Tetromino): 将被锁定的方块
    config (GameConfig): 游戏配置

    返回:
    tuple: (处理消行后的新网格, 消除的行数)
    """
    new_grid = [row[:] for row in grid]

    for row_idx, row in enumerate(piece.matrix):
        for col_idx, cell in enumerate(row):
            if cell != "X":
                continue

            board_x = piece.x + col_idx
            board_y = piece.y + row_idx
            if board_y >= 0:
                new_grid[board_y][board_x] = piece.shape_name

    kept_rows = [row for row in new_grid if 0 in row]
    lines_cleared = config.grid_rows - len(kept_rows)
    for _ in range(lines_cleared):
        kept_rows.insert(0, [0 for _ in range(config.grid_cols)])

    return kept_rows, lines_cleared


def board_profile(grid):
    """
    统计并返回网格当前各列的高度和总体空洞数，为启发式评估提供数据。

    参数:
    grid (list): 需要评估的网格矩阵

    返回:
    tuple: (列高列表, 空洞总数)
    """
    grid_rows = len(grid)
    grid_cols = len(grid[0]) if grid else 0
    heights = []
    holes = 0

    for col in range(grid_cols):
        first_block_row = None
        for row in range(grid_rows):
            if grid[row][col] != 0:
                first_block_row = row
                break

        if first_block_row is None:
            heights.append(0)
            continue

        heights.append(grid_rows - first_block_row)
        for row in range(first_block_row + 1, grid_rows):
            if grid[row][col] == 0:
                holes += 1

    return heights, holes


def evaluate_grid(grid, lines_cleared):
    """
    用启发式算法评估当前网格局面得分。
    - 奖励消行越多越好
    - 惩罚总堆高、最高列高度、产生的空洞和表面起伏
    
    参数:
    grid (list): 待评估网格
    lines_cleared (int): 该步操作消除的行数

    返回:
    float: 最终评估分数
    """
    heights, holes = board_profile(grid)
    aggregate_height = sum(heights)
    max_height = max(heights, default=0)
    bumpiness = sum(
        abs(left_height - right_height)
        for left_height, right_height in zip(heights, heights[1:])
    )

    return (
        lines_cleared * 1500
        - aggregate_height * 5.0
        - max_height * 18.0
        - holes * 56.0
        - bumpiness * 11.0
    )


def generate_candidates(grid, piece, config):
    """
    通过枚举该方块全部的旋转状态和横向位置，找出所有合法着陆点，并计算对应的启发式分数，将结果收集返回。

    参数:
    grid (list): 游戏场地网格
    piece (Tetromino): 当前活动方块
    config (GameConfig): 全局配置

    返回:
    list: 候选走法的字典列表
    """
    candidates = []

    for rotation_steps, matrix in unique_rotations(piece.matrix):
        min_x, max_x = valid_x_range(matrix, config.grid_cols)
        for target_x in range(min_x, max_x + 1):
            trial_piece = clone_piece(piece)
            trial_piece.matrix = [row[:] for row in matrix]
            trial_piece.x = target_x
            trial_piece.y = piece.y

            if trial_piece.check_collision(grid=grid):
                continue

            trial_piece.y = trial_piece.get_drop_position(grid)
            result_grid, lines_cleared = simulate_lock(grid, trial_piece, config)
            surface_score = evaluate_grid(result_grid, lines_cleared)
            candidates.append(
                {
                    "rotation_steps": rotation_steps,
                    "target_x": target_x,
                    "surface_score": surface_score,
                    "result_grid": result_grid,
                    "lines_cleared": lines_cleared,
                }
            )

    return candidates


def best_future_score(grid, next_piece, config):
    """
    计算基于下个方块可以带来的最高分，实现一层简单前瞻，避免陷入局部最优。

    参数:
    grid (list): 未来状态方块锁死后的场景网格
    next_piece (Tetromino): 预告的下一个方块
    config (GameConfig): 规则设定

    返回:
    float: 考虑后续方块的得分估计上限
    """
    if next_piece is None:
        return 0.0

    future_piece = clone_piece(next_piece)
    candidates = generate_candidates(grid, future_piece, config)
    if not candidates:
        return -9999.0

    return max(candidate["surface_score"] for candidate in candidates)


class AIController:
    """
    双模式 AI 控制器：
    1) heuristic: 启发式搜索与一层前瞻。
    2) model: 加载模型直接输出动作，失败时自动回退 heuristic。
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        action_interval_ms: int = 90,
        mistake_chance: float = 0.18,
        lookahead_weight: float = 0.28,
        mode: str | None = None,
        model_path: str | None = None,
    ):
        self.config = config if config is not None else CONFIG
        self.mode = (mode or self.config.ai_controller_mode or "heuristic").strip().lower()
        self.model_path = model_path or self.config.ai_model_path
        self.model_device = self.config.ai_model_device

        if self.mode == "model":
            action_interval_ms = self.config.ai_model_action_interval_ms
            mistake_chance = 0.0

        self.action_interval_ms = int(action_interval_ms)
        self.mistake_chance = float(mistake_chance)
        self.lookahead_weight = float(lookahead_weight)

        self._action_timer = 0
        self._planned_piece_id = None
        self._rotation_steps_remaining = 0
        self._target_x = 0
        self.last_plan_score = 0.0

        self._model = None
        self._model_error = ""
        self._pending_model_action = None
        self._pending_model_delay_ms = 0
        self._model_reaction_delay_ms = max(0, int(self.config.ai_model_reaction_delay_ms))
        if self.mode == "model":
            self._try_load_model()

    def set_profile(self, action_interval_ms: int, mistake_chance: float):
        if self.mode == "model":
            self.action_interval_ms = max(45, int(action_interval_ms))
            self.mistake_chance = 0.0
            return

        self.action_interval_ms = max(35, int(action_interval_ms))
        self.mistake_chance = max(0.0, min(0.4, float(mistake_chance)))

    def reset(self):
        self._action_timer = 0
        self._planned_piece_id = None
        self._rotation_steps_remaining = 0
        self._target_x = 0
        self.last_plan_score = 0.0
        self._pending_model_action = None
        self._pending_model_delay_ms = 0

    def update(self, game_core, dt):
        if game_core.state != "RUNNING" or game_core.current_piece is None:
            return

        if self.mode == "model" and self._model is not None:
            self._update_with_model(game_core, dt)
            return

        if self._planned_piece_id != id(game_core.current_piece):
            self._plan_for_piece(game_core)

        self._action_timer += dt
        if self._action_timer < self.action_interval_ms:
            return

        self._action_timer -= self.action_interval_ms
        self._execute_next_action(game_core)

    def _try_load_model(self):
        if torch is None:
            self._model_error = "torch 不可用，自动回退启发式模式"
            self.mode = "heuristic"
            return

        model_file = Path(self.model_path)
        if not model_file.exists():
            self._model_error = f"模型文件不存在: {self.model_path}"
            self.mode = "heuristic"
            return

        try:
            device = torch.device(self.model_device)
            self._model = torch.jit.load(str(model_file), map_location=device)
            self._model.eval()
            self._model_error = ""
            return
        except Exception:
            pass

        try:
            loaded = torch.load(str(model_file), map_location=self.model_device)
            if hasattr(loaded, "eval"):
                loaded.eval()
                self._model = loaded
                self._model_error = ""
                return
            self._model_error = "模型格式不支持，自动回退启发式模式"
            self.mode = "heuristic"
        except Exception as exc:
            self._model_error = f"模型加载失败: {exc}"
            self.mode = "heuristic"

    def _build_model_state(self, game_core):
        heights, holes = board_profile(game_core.grid)
        aggregate_height = sum(heights)
        max_height = max(heights, default=0)
        bumpiness = sum(
            abs(left_height - right_height)
            for left_height, right_height in zip(heights, heights[1:])
        )

        def _piece_onehot(piece):
            vec = [0.0] * 7
            order = ["I", "J", "L", "O", "S", "T", "Z"]
            if piece is None:
                return vec
            if piece.shape_name in order:
                vec[order.index(piece.shape_name)] = 1.0
            return vec

        total_cells = max(1, self.config.grid_rows * self.config.grid_cols)
        state = [
            aggregate_height / max(1, self.config.grid_cols * self.config.grid_rows),
            max_height / max(1, self.config.grid_rows),
            holes / total_cells,
            bumpiness / max(1, self.config.grid_cols),
            min(1.0, game_core.lines_cleared_total / 200.0),
        ]
        state.extend(_piece_onehot(game_core.current_piece))
        state.extend(_piece_onehot(game_core.next_piece))

        incoming_count = float(len(getattr(game_core, "incoming_garbage", [])))
        state.append(min(1.0, incoming_count / 20.0))
        return state

    def _select_model_action(self, game_core):
        if self._model is None or torch is None:
            return None

        state = self._build_model_state(game_core)
        tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        try:
            with torch.no_grad():
                output = self._model(tensor)
        except Exception:
            return None

        if hasattr(output, "detach"):
            logits = output.detach().flatten().tolist()
        elif isinstance(output, (list, tuple)):
            logits = list(output)
        else:
            return None

        if len(logits) < 5:
            return None

        return int(max(range(5), key=lambda idx: logits[idx]))

    def _execute_model_action(self, game_core, action_idx: int):
        piece = game_core.current_piece
        if piece is None:
            return

        if action_idx == 0:
            if not piece.check_collision(dx=-1, dy=0, grid=game_core.grid):
                piece.x -= 1
            return

        if action_idx == 1:
            if not piece.check_collision(dx=1, dy=0, grid=game_core.grid):
                piece.x += 1
            return

        if action_idx == 2:
            piece.rotate(game_core.grid)
            return

        if action_idx == 3:
            if not piece.check_collision(dx=0, dy=1, grid=game_core.grid):
                piece.y += 1
            else:
                game_core.lock_shape()
            return

        piece.y = piece.get_drop_position(game_core.grid)
        game_core.lock_shape()

    def _update_with_model(self, game_core, dt):
        self._action_timer += dt
        if self._action_timer < self.action_interval_ms:
            return

        self._action_timer -= self.action_interval_ms
        if self._pending_model_action is None:
            self._pending_model_action = self._select_model_action(game_core)
            self._pending_model_delay_ms = 0

        if self._pending_model_action is None:
            if self._planned_piece_id != id(game_core.current_piece):
                self._plan_for_piece(game_core)
            self._execute_next_action(game_core)
            return

        self._pending_model_delay_ms += self.action_interval_ms
        if self._pending_model_delay_ms < self._model_reaction_delay_ms:
            return

        action = self._pending_model_action
        self._pending_model_action = None
        self._pending_model_delay_ms = 0
        self._execute_model_action(game_core, action)

    def _plan_for_piece(self, game_core):
        piece = game_core.current_piece
        candidates = generate_candidates(game_core.grid, piece, self.config)

        if not candidates:
            self._rotation_steps_remaining = 0
            self._target_x = piece.x
            self._planned_piece_id = id(piece)
            self.last_plan_score = -9999.0
            return

        for candidate in candidates:
            future_score = best_future_score(
                candidate["result_grid"],
                game_core.next_piece,
                self.config,
            )
            target_bias = abs(candidate["target_x"] - self.config.grid_cols // 2) * 0.3
            candidate["total_score"] = (
                candidate["surface_score"]
                + future_score * self.lookahead_weight
                - target_bias
            )

        candidates.sort(key=lambda item: item["total_score"], reverse=True)
        selected = candidates[0]

        if len(candidates) > 1 and random.random() < self.mistake_chance:
            selected = random.choice(candidates[: min(3, len(candidates))])

        self._rotation_steps_remaining = selected["rotation_steps"]
        self._target_x = selected["target_x"]
        self._planned_piece_id = id(piece)
        self.last_plan_score = selected["total_score"]

    def _execute_next_action(self, game_core):
        piece = game_core.current_piece

        if self._rotation_steps_remaining > 0:
            if piece.rotate(game_core.grid):
                self._rotation_steps_remaining -= 1
            else:
                self._plan_for_piece(game_core)
            return

        if piece.x < self._target_x:
            if not piece.check_collision(dx=1, dy=0, grid=game_core.grid):
                piece.x += 1
            else:
                self._plan_for_piece(game_core)
            return

        if piece.x > self._target_x:
            if not piece.check_collision(dx=-1, dy=0, grid=game_core.grid):
                piece.x -= 1
            else:
                self._plan_for_piece(game_core)
            return

        piece.y = piece.get_drop_position(game_core.grid)
        game_core.lock_shape()
        self._planned_piece_id = None
