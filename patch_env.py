import re

with open("tetris_env.py", "r") as f:
    content = f.read()

new_methods = """    def get_next_states(self) -> dict:
        \"\"\"
        获取当前方块穷举的所有合法落座状态。
        返回: dict{(rotation_steps, target_x): state_vector}
        \"\"\"
        from ai_controller import generate_candidates
        states = {}
        
        piece = self.player_core.current_piece
        if not piece:
            return states
            
        candidates = generate_candidates(self.player_core.grid, piece, self.config)
        for cand in candidates:
            rot = cand["rotation_steps"]
            tx = cand["target_x"]
            sim_grid = cand["result_grid"]
            
            # 使用专门的特征提取
            state_vec = self._extract_board_features(sim_grid)
            states[(rot, tx)] = state_vec
            
        return states

    def _extract_board_features(self, grid) -> list[float]:
        cols = self.config.grid_cols
        rows = self.config.grid_rows
        heights = []
        holes = 0
        for x in range(cols):
            col_height = 0
            col_holes = 0
            found = False
            for y in range(rows):
                if grid[y][x]:
                    if not found:
                        col_height = rows - y
                        found = True
                else:
                    if found:
                        col_holes += 1
            heights.append(col_height)
            holes += col_holes
            
        state = []
        state.extend([h for h in heights])
        state.append(holes)
        bumpiness = sum(abs(heights[i] - heights[i-1]) for i in range(1, cols))
        state.append(bumpiness)
        
        # 补齐原本的网络输入维度 (坐标和陷阱权) - 对于评估落点其实这些不变
        p_piece = self.player_core.current_piece
        if p_piece:
            state.extend([p_piece.x, p_piece.y])
        else:
            state.extend([0, 0])
            
        ai_piece = self.ai_core.current_piece
        if ai_piece:
            state.extend([ai_piece.x, ai_piece.y])
        else:
            state.extend([0, 0])
            
        can_trap = 1.0 if (self.player_trap_energy >= self.config.versus_trap_energy_cost and self.player_trap_cooldown_ms <= 0) else 0.0
        state.append(can_trap)
        
        return state

    def step_next_state(self, rotation_steps: int, target_x: int):
        \"\"\"一键抵达目标状态的跃迁动作\"\"\"
        if self._is_done():
            return self.step(ACTION_NOOP)
            
        self.episode_step += 1
        self._recent_events = []
        
        self._tick_battle_states(self.step_dt_ms)
        
        piece = self.player_core.current_piece
        if piece:
            for _ in range(rotation_steps):
                piece.rotate(self.player_core.grid)
            piece.x = target_x
            piece.y = piece.get_drop_position(self.player_core.grid)
            self._recent_player_piece_y = piece.y
            self.player_core.lock_shape()
            
        self.ai_controller.update(self.ai_core, self.step_dt_ms)
        self.player_core.update(self.step_dt_ms)
        self.ai_core.update(self.step_dt_ms)
        
        self._process_lock_events()
        self._try_ai_activate_trap()
        
        reward = self._compute_reward()
        done = self._is_done()
        info = self._build_info(action=ACTION_HARD_DROP, reward=reward, done=done)
        
        self._sync_prev_metrics()
        
        # state vector 其实应该传真实的当前盘面，我们可以复用刚才的 _extract
        return StepOutcome(
            state=self._get_state_vector(),
            reward=reward,
            done=done,
            info=info
        )"""

# Insert before def _is_done(self) -> bool:
content = content.replace("    def _is_done(self) -> bool:", new_methods + "\n\n    def _is_done(self) -> bool:")

with open("tetris_env.py", "w") as f:
    f.write(content)
print("Environment patched")
