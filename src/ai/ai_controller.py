import random

from src.ai.dqn_model import load_inference_model
from src.ai.next_state_features import analyze_board, extract_next_state_features
from settings import CONFIG, GameConfig
from src.game.tetromino import Tetromino

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
    - 特别惩罚底层空洞（难以消除）
    
    参数:
    grid (list): 待评估网格
    lines_cleared (int): 该步操作消除的行数

    返回:
    float: 最终评估分数
    """
    stats = analyze_board(grid)
    heights = stats["heights"]
    holes = float(stats["holes"])
    aggregate_height = float(stats["aggregate_height"])
    max_height = float(stats["max_height"])
    covered_holes = float(stats["covered_holes"])
    danger_cells = float(stats["danger_cells"])
    # 新增：深层空洞
    deep_holes = float(stats.get("deep_holes", 0.0))
    deep_hole_depth = float(stats.get("deep_hole_depth_sum", 0.0))
    rows = max(1.0, float(stats["rows"]))
    bumpiness = sum(
        abs(left_height - right_height)
        for left_height, right_height in zip(heights, heights[1:])
    )
    split = max(1, len(heights) // 2)
    left_height_sum = sum(heights[:split])
    right_height_sum = sum(heights[split:])
    side_imbalance = abs(left_height_sum - right_height_sum)
    left_peak = max(heights[:split], default=0)
    right_peak = max(heights[split:], default=0)
    peak_imbalance = abs(left_peak - right_peak)
    near_top_excess = sum(max(0.0, float(height) - (rows - 6.0)) for height in heights)
    high_columns = sum(1 for height in heights if float(height) >= rows - 4.0)
    single_line_bias = 220.0 if lines_cleared == 1 else 0.0

    return (
        lines_cleared * 1150.0
        - single_line_bias
        - aggregate_height * 6.5
        - max_height * 24.0
        - holes * 64.0
        - covered_holes * 1.8
        - bumpiness * 12.5
        - side_imbalance * 4.5
        - peak_imbalance * 10.0
        - danger_cells * 95.0
        - near_top_excess * 42.0
        - high_columns * 28.0
        # 新增：深层空洞惩罚（底部空洞更难消除，惩罚更重）
        - deep_holes * 120.0
        - deep_hole_depth * 80.0
    )


def risk_penalty(grid):
    stats = analyze_board(grid)
    heights = stats["heights"]
    rows = max(1.0, float(stats["rows"]))
    danger_cells = float(stats["danger_cells"])
    near_top_excess = sum(max(0.0, float(height) - (rows - 6.0)) for height in heights)
    high_columns = sum(1 for height in heights if float(height) >= rows - 4.0)
    max_height = float(stats["max_height"])

    return (
        danger_cells * 24.0
        + near_top_excess * 12.0
        + high_columns * 8.0
        + max(0.0, max_height - (rows - 5.0)) * 10.0
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
        trial_piece = clone_piece(piece)
        trial_piece.matrix = [row[:] for row in matrix]
        spawn_y = piece.y
        for target_x in range(min_x, max_x + 1):
            trial_piece.x = target_x
            trial_piece.y = spawn_y

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
        self._planned_garbage_received_total = 0
        self._planned_incoming_garbage_len = 0
        self._closed_loop_enabled = self.mode == "model"

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
        self._clear_plan()
        self.last_plan_score = 0.0
        self._pending_model_action = None
        self._pending_model_delay_ms = 0

    def _clear_plan(self):
        self._planned_piece_id = None
        self._rotation_steps_remaining = 0
        self._target_x = 0
        self._planned_garbage_received_total = 0
        self._planned_incoming_garbage_len = 0

    def _mark_planned_snapshot(self, game_core):
        self._planned_garbage_received_total = int(getattr(game_core, "garbage_received_total", 0))
        self._planned_incoming_garbage_len = int(len(getattr(game_core, "incoming_garbage", [])))

    def _should_recalculate_plan(self, game_core) -> bool:
        if not self._closed_loop_enabled:
            return False

        current_received = int(getattr(game_core, "garbage_received_total", 0))
        current_incoming = int(len(getattr(game_core, "incoming_garbage", [])))
        if current_received > self._planned_garbage_received_total:
            return True
        if current_incoming != self._planned_incoming_garbage_len:
            return True
        return False

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

        self._model, self._model_error = load_inference_model(
            self.model_path,
            map_location=self.model_device,
            expected_input_dim=20,
            expected_action_dim=5,
        )
        if self._model is None:
            self.mode = "heuristic"
            if not self._model_error:
                self._model_error = "模型格式不支持，自动回退启发式模式"

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

    def _select_best_candidate(self, game_core, allow_mistake: bool = True):
        piece = game_core.current_piece
        candidates = generate_candidates(game_core.grid, piece, self.config)
        if not candidates:
            return None

        if self._model is not None and torch is not None:
            can_trap = False
            features = [
                extract_next_state_features(
                    candidate["result_grid"],
                    player_piece=game_core.current_piece,
                    ai_piece=game_core.next_piece,
                    can_trap=can_trap,
                    lines_cleared=float(candidate.get("lines_cleared", 0)),
                )
                for candidate in candidates
            ]
            tensor = torch.as_tensor(features, dtype=torch.float32)
            try:
                model_device = next(self._model.parameters()).device
                tensor = tensor.to(model_device)
                with torch.no_grad():
                    scores = self._model(tensor).squeeze(-1).detach().cpu().tolist()
                for candidate, score in zip(candidates, scores):
                    candidate["surface_score"] = float(score)
            except Exception:
                pass

        for candidate in candidates:
            if self.lookahead_weight > 0.0:
                future_score = best_future_score(
                    candidate["result_grid"],
                    game_core.next_piece,
                    self.config,
                )
            else:
                future_score = 0.0
            stack_risk = risk_penalty(candidate["result_grid"])
            target_bias = abs(candidate["target_x"] - self.config.grid_cols // 2) * 0.3
            candidate["total_score"] = (
                candidate["surface_score"]
                + future_score * self.lookahead_weight
                - target_bias
                - stack_risk
            )

        candidates.sort(key=lambda item: item["total_score"], reverse=True)
        selected = candidates[0]

        if allow_mistake and len(candidates) > 1 and random.random() < self.mistake_chance:
            selected = random.choice(candidates[: min(3, len(candidates))])

        return selected

    def _plan_for_piece(self, game_core):
        piece = game_core.current_piece
        selected = self._select_best_candidate(game_core, allow_mistake=True)
        if selected is None:
            self._rotation_steps_remaining = 0
            self._target_x = piece.x
            self._planned_piece_id = id(piece)
            self.last_plan_score = -9999.0
            self._mark_planned_snapshot(game_core)
            return

        self._rotation_steps_remaining = int(selected["rotation_steps"])
        self._target_x = int(selected["target_x"])
        self._planned_piece_id = id(piece)
        self.last_plan_score = float(selected["total_score"])
        self._mark_planned_snapshot(game_core)

    def _execute_next_action(self, game_core):
        piece = game_core.current_piece
        if piece is None:
            self._clear_plan()
            return

        if self._planned_piece_id != id(piece) or self._should_recalculate_plan(game_core):
            self._plan_for_piece(game_core)

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

        if self._closed_loop_enabled:
            # Final check before hard-drop: re-evaluate best move on the real current board.
            # If target changed, update plan and defer hard drop to next tick.
            reselected = self._select_best_candidate(game_core, allow_mistake=False)
            if reselected is not None:
                new_target_x = int(reselected["target_x"])
                new_rot_steps = int(reselected["rotation_steps"])
                if new_target_x != int(piece.x) or new_rot_steps != 0:
                    self._target_x = new_target_x
                    self._rotation_steps_remaining = new_rot_steps
                    self._planned_piece_id = id(piece)
                    self.last_plan_score = float(reselected["total_score"])
                    self._mark_planned_snapshot(game_core)
                    return

        piece.y = piece.get_drop_position(game_core.grid)
        game_core.lock_shape()
        self._clear_plan()
