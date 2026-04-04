import re

with open("tetris_env.py", "r") as f:
    text = f.read()

# Let's adjust reward to penalize holes more heavily because that's usually why games end in 60-100 lines.
# And reward clear lines quadratically instead of linearly? Or rather, just increase penalty for holes and bumpiness.
# Let's check _compute_reward
old_holes_penalty = "reward -= delta_holes * 1.5"
new_holes_penalty = "reward -= delta_holes * 3.5"
text = text.replace(old_holes_penalty, new_holes_penalty)

old_bumpiness = "reward -= delta_bumpiness * 0.2"
new_bumpiness = "reward -= delta_bumpiness * 0.5"
text = text.replace(old_bumpiness, new_bumpiness)

with open("tetris_env.py", "w") as f:
    f.write(text)

