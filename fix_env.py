import re

with open("tetris_env.py", "r") as f:
    text = f.read()

old_code = """        piece = self.player_core.current_piece
        if piece:
            for _ in range(rotation_steps):
                piece.rotate(self.player_core.grid)
            piece.x = target_x
            piece.y = piece.get_drop_position(self.player_core.grid)"""

new_code = """        piece = self.player_core.current_piece
        if piece:
            # 必须绕过碰撞检测强制旋转矩阵，否则如果在出生点旋转由于空间不足会被拒绝旋转
            for _ in range(rotation_steps):
                piece.matrix = piece.get_rotated_matrix()
            piece.x = target_x
            piece.y = piece.get_drop_position(self.player_core.grid)"""

text = text.replace(old_code, new_code)

with open("tetris_env.py", "w") as f:
    f.write(text)

