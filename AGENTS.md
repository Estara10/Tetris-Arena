# Repository Guidelines

## Project Structure & Module Organization

This repository is a Pygame Tetris AI-versus project. `main.py` is the entry point and delegates to `app_controller.py` for menu and match orchestration. Core gameplay lives in `game_core.py`, `tetromino.py`, `piece_sequence.py`, and `player_input.py`. Rendering and UI are handled by `render.py`, `menu_screens.py`, `background.py`, and `ui_fonts.py`.

Match logic is split between `versus_match.py` for two-board player-vs-AI play and `shared_arena_match.py` / `shared_game_core.py` for same-board arena play. AI and training code lives in `ai_controller.py`, `tetris_env.py`, `dqn_model.py`, `deep_q_network.py`, `rl_trainer.py`, and `train_nextstate.py`. Assets and model outputs are stored in `background.png`, `codioful.jpg`, and `models/`.

## Build, Test, and Development Commands

- `python3 main.py`: run the interactive game.
- `python3 -m py_compile main.py render.py game_core.py settings.py ai_controller.py tetromino.py background.py`: quick syntax check for core runtime modules.
- `python3 test_shared_load.py`: smoke-test shared arena imports.
- `python3 test_load.py`: try loading the next-state model checkpoint.
- `python3 train_nextstate.py --episodes 100 --device cpu`: run a short CPU training pass.
- `python3 evaluate_model.py --mode TRADITIONAL --episodes 5 --device cpu`: evaluate a checkpoint.

Install runtime dependencies manually as needed, typically `pygame` for gameplay and `torch` for model training/inference.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation. Prefer clear module-level functions and small classes that match the current style. Use `snake_case` for functions, variables, and files; use `PascalCase` for classes such as `GameCore` and `SharedArenaMatch`. Keep configuration in `settings.py` rather than scattering constants. Preserve Chinese UI text where it is user-facing.

## Testing Guidelines

There is no formal test framework yet; use smoke tests and targeted scripts. Add lightweight `test_*.py` files for import checks, model-loading checks, or deterministic logic checks. For gameplay changes, at minimum run `py_compile` and the relevant smoke test. For AI/training changes, run a short training or evaluation command on CPU before submitting.

## Commit & Pull Request Guidelines

Recent commits use short Chinese summaries, for example `优化AI逻辑` and `更新Q-learning`. Keep commits concise and action-oriented. Pull requests should describe the gameplay or training behavior changed, list commands run, and include screenshots or short recordings for UI changes. Do not commit `__pycache__/`, generated `.pyc` files, or large new model checkpoints unless they are intentionally part of the change.

## Configuration & Assets

Environment variables in `settings.py` control AI mode, model paths, training parameters, and arena timing. Examples: `TETRIS_AI_MODE=model`, `TETRIS_AI_MODEL_PATH=models/next_state_v3/latest_checkpoint.pth`, and `TETRIS_ARENA_FALL_MS=500`.
