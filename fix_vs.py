import re

with open("versus_match.py", "r") as f:
    text = f.read()

# Fix the manual hardcode overrides
text = text.replace('mode="heuristic",#暂时只用启发式AI --- IGNORE ---', 'mode="model",')
text = text.replace('model_path=self.config.ai_model_path,', 'model_path="models/next_state/best.pt",')

with open("versus_match.py", "w") as f:
    f.write(text)

with open("shared_arena_match.py", "r") as f:
    text2 = f.read()

text2 = text2.replace('mode="heuristic",', 'mode="model",')
text2 = text2.replace('model_path=resolve_existing_model_path("models", "shared_arena"),', 'model_path="models/next_state/best.pt",')

with open("shared_arena_match.py", "w") as f:
    f.write(text2)

