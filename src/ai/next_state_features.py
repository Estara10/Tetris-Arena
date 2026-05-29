from __future__ import annotations


NEXT_STATE_AUX_FEATURE_COUNT = 18


def next_state_feature_size(grid_cols: int) -> int:
    return max(1, int(grid_cols)) + NEXT_STATE_AUX_FEATURE_COUNT


def analyze_board(grid, danger_rows: int | None = None) -> dict[str, float | list[int]]:
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    danger_band = max(1, min(rows, danger_rows if danger_rows is not None else max(4, rows // 5)))

    heights: list[int] = []
    holes = 0
    covered_holes = 0
    aggregate_height = 0
    
    # 新增：深层空洞统计（底部1/3区域的空洞更严重）
    deep_holes = 0  # 底部区域的空洞数
    deep_hole_depth_sum = 0  # 空洞深度加权和（越深惩罚越重）
    bottom_third = rows * 2 // 3  # 底部1/3的起始行

    for x in range(cols):
        col_height = 0
        blocks_seen = 0
        for y in range(rows):
            filled = grid[y][x] != 0
            if filled:
                blocks_seen += 1
                if col_height == 0:
                    col_height = rows - y
            elif blocks_seen > 0:
                holes += 1
                covered_holes += blocks_seen
                # 统计深层空洞
                if y >= bottom_third:
                    deep_holes += 1
                    # 越靠近底部，惩罚越重（深度权重）
                    depth_weight = (y - bottom_third + 1) / max(1, rows - bottom_third)
                    deep_hole_depth_sum += depth_weight
        heights.append(col_height)
        aggregate_height += col_height

    max_height = max(heights, default=0)
    bumpiness = sum(abs(heights[i] - heights[i - 1]) for i in range(1, cols))

    row_transitions = 0
    for row in grid:
        prev_filled = True
        for cell in row:
            filled = cell != 0
            if filled != prev_filled:
                row_transitions += 1
            prev_filled = filled
        if not prev_filled:
            row_transitions += 1

    col_transitions = 0
    for x in range(cols):
        prev_filled = True
        for y in range(rows):
            filled = grid[y][x] != 0
            if filled != prev_filled:
                col_transitions += 1
            prev_filled = filled
        if not prev_filled:
            col_transitions += 1

    wells = 0
    for x in range(cols):
        depth = 0
        for y in range(rows):
            filled = grid[y][x] != 0
            left_filled = x == 0 or grid[y][x - 1] != 0
            right_filled = x == cols - 1 or grid[y][x + 1] != 0
            if (not filled) and left_filled and right_filled:
                depth += 1
                wells += depth
            else:
                depth = 0

    danger_cells = 0
    for y in range(danger_band):
        for x in range(cols):
            if grid[y][x] != 0:
                danger_cells += 1

    return {
        "rows": rows,
        "cols": cols,
        "heights": heights,
        "holes": float(holes),
        "covered_holes": float(covered_holes),
        "aggregate_height": float(aggregate_height),
        "max_height": float(max_height),
        "bumpiness": float(bumpiness),
        "row_transitions": float(row_transitions),
        "col_transitions": float(col_transitions),
        "wells": float(wells),
        "danger_cells": float(danger_cells),
        "danger_rows": float(danger_band),
        "deep_holes": float(deep_holes),
        "deep_hole_depth_sum": float(deep_hole_depth_sum),
        "bottom_third_row": float(bottom_third),
    }


def _resolve_position(piece=None, pos=None) -> tuple[float, float]:
    if pos is not None:
        return float(pos[0]), float(pos[1])
    if piece is not None:
        return float(getattr(piece, "x", 0.0)), float(getattr(piece, "y", 0.0))
    return 0.0, 0.0


def extract_next_state_features(
    grid,
    *,
    player_piece=None,
    ai_piece=None,
    player_pos=None,
    ai_pos=None,
    can_trap: bool = False,
    lines_cleared: float = 0.0,
) -> list[float]:
    stats = analyze_board(grid)
    rows = max(1, int(stats["rows"]))
    cols = max(1, int(stats["cols"]))
    total_cells = rows * cols

    player_x, player_y = _resolve_position(piece=player_piece, pos=player_pos)
    ai_x, ai_y = _resolve_position(piece=ai_piece, pos=ai_pos)

    state = [float(height) / float(rows) for height in stats["heights"]]
    state.append(float(stats["aggregate_height"]) / float(total_cells))
    state.append(float(stats["max_height"]) / float(rows))
    state.append(float(stats["holes"]) / float(total_cells))
    state.append(float(stats["covered_holes"]) / float(total_cells * rows))
    state.append(float(stats["bumpiness"]) / float(max(1, (cols - 1) * rows)))
    state.append(float(stats["row_transitions"]) / float(max(1, rows * (cols + 1))))
    state.append(float(stats["col_transitions"]) / float(max(1, cols * (rows + 1))))
    state.append(float(stats["wells"]) / float(total_cells))
    state.append(
        float(stats["danger_cells"])
        / float(max(1, int(stats["danger_rows"]) * cols))
    )
    # 新增：深层空洞特征（底部空洞对消行影响更大）
    bottom_cells = max(1, (rows - int(stats["bottom_third_row"])) * cols)
    state.append(float(stats["deep_holes"]) / float(bottom_cells))
    state.append(float(stats["deep_hole_depth_sum"]) / float(max(1, cols)))
    # 新增：底部填充率（底部越满越好，说明可以消行）
    bottom_filled = 0
    bottom_row_start = int(stats["bottom_third_row"])
    for y in range(bottom_row_start, rows):
        for x in range(cols):
            if grid[y][x] != 0:
                bottom_filled += 1
    state.append(float(bottom_filled) / float(bottom_cells))
    
    state.append(player_x / float(max(1, cols - 1)))
    state.append(player_y / float(max(1, rows - 1)))
    state.append(ai_x / float(max(1, cols - 1)))
    state.append(ai_y / float(max(1, rows - 1)))
    state.append(1.0 if can_trap else 0.0)
    state.append(max(0.0, min(1.0, float(lines_cleared) / 4.0)))
    return state
