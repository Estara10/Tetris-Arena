import re

with open("tetris_env.py", "r") as f:
    content = f.read()

# 1. get_next_states
o1 = """            sim_grid = cand["result_grid"]
            
            # 使用专门的特征提取
            state_vec = self._extract_board_features(sim_grid)"""
n1 = """            sim_grid = cand["result_grid"]
            lines_cleared = cand.get("lines_cleared", 0)
            
            # 使用专门的特征提取
            state_vec = self._extract_board_features(sim_grid, lines_cleared)"""
content = content.replace(o1, n1)

# 2. _extract_board_features signature
o2 = """    def _extract_board_features(self, grid) -> list[float]:"""
n2 = """    def _extract_board_features(self, grid, lines_cleared=0) -> list[float]:"""
content = content.replace(o2, n2)

# 3. _extract appending lines_cleared
o3 = """        state.append(can_trap)
        
        return state"""
n3 = """        state.append(can_trap)
        
        # 将本次动作消除的行数额外放入特征中
        state.append(float(lines_cleared))
        
        return state"""
content = content.replace(o3, n3)

# 4. _get_state_vector appending dummy
o4 = """        state.append(can_trap)

        return state
    def _board_binary(self, grid: list[list[int | str]]) -> list[float]:"""
n4 = """        state.append(can_trap)
        
        # padding feature for regular state queries to match dimension
        state.append(0.0)

        return state
    def _board_binary(self, grid: list[list[int | str]]) -> list[float]:"""
content = content.replace(o4, n4)

with open("tetris_env.py", "w") as f:
    f.write(content)

