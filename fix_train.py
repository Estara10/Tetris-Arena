import re

with open("train_nextstate.py", "r") as f:
    text = f.read()

old_code = """    env = TetrisEnv()
    model = DeepQNetwork().to(device)"""

new_code = """    env = TetrisEnv()
    state_dim = len(env._get_state_vector())
    model = DeepQNetwork(input_dim=state_dim).to(device)"""

text = text.replace(old_code, new_code)

with open("train_nextstate.py", "w") as f:
    f.write(text)

