import re

with open("ai_controller.py", "r") as f:
    text = f.read()

# Enhance heuristic baseline evaluation matching default DQN input to prevent AI from getting stuck doing stupid moves
# if it learns a bad mode early on.

