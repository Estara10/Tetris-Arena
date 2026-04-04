from __future__ import annotations

from pathlib import Path


DEFAULT_MODEL_CANDIDATES = (
    "latest_checkpoint.pth",
    "last.pt",
    "best.pt",
)


def _normalize_mode_key(mode_key: str) -> str:
    return str(mode_key or "").strip().lower()


def mode_model_dir(base_dir: str | Path, mode_key: str) -> Path:
    return Path(base_dir) / _normalize_mode_key(mode_key)


def recommended_model_path(base_dir: str | Path, mode_key: str, filename: str = "best.pt") -> Path:
    return mode_model_dir(base_dir, mode_key) / filename


def _latest_dated_best(search_dir: Path) -> Path | None:
    if not search_dir.exists() or not search_dir.is_dir():
        return None

    dated_files = [
        path
        for path in search_dir.glob("best_*.pt")
        if path.is_file() and path.stem[5:].isdigit()
    ]
    if not dated_files:
        return None

    return max(dated_files, key=lambda p: p.name)


def resolve_existing_model_path(
    base_dir: str | Path,
    mode_key: str,
    filename: str | None = None,
) -> Path:
    import os
    base_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), base_dir)))
    """
    解析可用模型路径：
    - 指定 filename 时按该文件查找；
    - 未指定时优先 latest_checkpoint.pth，其次 last.pt，再次 best.pt。
    """
    candidates = (filename,) if filename else DEFAULT_MODEL_CANDIDATES

    normalized_mode = _normalize_mode_key(mode_key)
    for name in candidates:
        if name == "best.pt":
            dated_mode_best = _latest_dated_best(mode_model_dir(base_dir, mode_key))
            if dated_mode_best is not None:
                return dated_mode_best

            if normalized_mode in {"classic", "traditional"}:
                dated_legacy_best = _latest_dated_best(Path(base_dir))
                if dated_legacy_best is not None:
                    return dated_legacy_best

        mode_path = recommended_model_path(base_dir, mode_key, filename=name)
        if mode_path.exists():
            return mode_path

        legacy_path = Path(base_dir) / name
        if normalized_mode in {"classic", "traditional"} and legacy_path.exists():
            return legacy_path

    first_name = candidates[0] if candidates else "best.pt"
    return recommended_model_path(base_dir, mode_key, filename=first_name)
