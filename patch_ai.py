import re

with open("ai_controller.py", "r") as f:
    text = f.read()

# completely replace _update_with_model and _select_model_action
# It's better to just hook into _plan_for_piece: if self._model is not None, use the model to evaluate surface_score.
# But first, we modify update(self) to always go to:
# def update(...): self._update_heuristic(game_core, dt) since planning logic works perfectly for both.

old_update = """    def update(self, game_core, dt):
        if self.ai_level == AILevel.HEURISTIC:
            self._update_heuristic(game_core, dt)
        else:
            self._update_with_model(game_core, dt)"""

new_update = """    def update(self, game_core, dt):
        self._update_heuristic(game_core, dt)"""

text = text.replace(old_update, new_update)

old_plan = """    def _plan_for_piece(self, game_core):
        piece = game_core.current_piece
        candidates = generate_candidates(game_core.grid, piece, self.config)"""

new_plan = """    def _plan_for_piece(self, game_core):
        piece = game_core.current_piece
        candidates = generate_candidates(game_core.grid, piece, self.config)

        if self._model is not None:
            # 使用深度学习模型评估网格
            from tetris_env import TetrisEnv
            env = TetrisEnv(mode="TRADITIONAL")
            env.player_core = game_core
            
            for candidate in candidates:
                state = env._extract_board_features(candidate["result_grid"])
                tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                try:
                    with torch.no_grad():
                        score = self._model(tensor).item()
                        candidate["surface_score"] = score
                except Exception:
                    pass
"""

text = text.replace(old_plan, new_plan)

with open("ai_controller.py", "w") as f:
    f.write(text)

print("AI patched.")
