from __future__ import annotations


ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE = 3
ACTION_SOFT_DROP = 4
ACTION_HARD_DROP = 5

ACTION_MEANING = {
    ACTION_NOOP: "noop",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
    ACTION_ROTATE: "rotate",
    ACTION_SOFT_DROP: "soft_drop",
    ACTION_HARD_DROP: "hard_drop",
}

PIECE_ORDER = ["I", "J", "L", "O", "S", "T", "Z"]


def piece_features(piece, grid_cols: int, grid_rows: int) -> list[float]:
    if piece is None:
        return [0.0] * 9

    values = [0.0] * len(PIECE_ORDER)
    if piece.shape_name in PIECE_ORDER:
        values[PIECE_ORDER.index(piece.shape_name)] = 1.0

    values.append(piece.x / float(grid_cols))
    values.append(piece.y / float(grid_rows))
    return values


def build_shared_arena_state(match, controlled_entity_id: str) -> list[float]:
    controlled = match._get_entity(controlled_entity_id)
    others = [ent for ent in match.entities if ent.id != controlled_entity_id]

    state: list[float] = []
    for row in match.core.grid:
        for cell in row:
            state.append(0.0 if cell == 0 else 1.0)

    ordered_entities = []
    if controlled is not None:
        ordered_entities.append(controlled)
    ordered_entities.extend(others)

    for ent in ordered_entities:
        state.extend(
            piece_features(
                getattr(ent, "piece", None),
                grid_cols=match.arena_config.grid_cols,
                grid_rows=match.arena_config.grid_rows,
            )
        )

    return state


def shared_arena_state_size(grid_rows: int, grid_cols: int, entity_count: int) -> int:
    return grid_rows * grid_cols + entity_count * 9
