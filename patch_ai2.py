import re

with open("ai_controller.py", "r") as f:
    text = f.read()

old_tensor = """                tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)"""
new_tensor = """                
                # We also need to mock ai_core and energies since feature extraction asks for it
                env.ai_core = game_core # Just mock
                env.player_trap_energy = 0
                env.player_trap_cooldown_ms = 0
                state = env._extract_board_features(candidate["result_grid"])
                tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)"""

text = text.replace(old_tensor, new_tensor)

with open("ai_controller.py", "w") as f:
    f.write(text)

print("AI patched 2.")
