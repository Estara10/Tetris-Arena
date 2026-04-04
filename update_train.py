import re

with open("train_nextstate.py", "r") as f:
    text = f.read()

# We need to add prioritized experience replay or better learning rate schedule
# Let's adjust reward calculation and add gradient clipping.
old_optim = "optimizer.step()"
new_optim = "nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)\n                optimizer.step()"
text = text.replace(old_optim, new_optim)

# Save the updated training script
with open("train_nextstate.py", "w") as f:
    f.write(text)

