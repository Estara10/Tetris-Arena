import argparse
import random
import torch
import torch.nn as nn
from collections import deque
from copy import deepcopy

from src.ai.tetris_env import TetrisEnv
from src.ai.deep_q_network import DeepQNetwork

def train(episodes=2000, batch_size=512, gamma=0.99, epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=2000):
    env = TetrisEnv(mode="TRADITIONAL")
    model = DeepQNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    memory = deque(maxlen=30000)
    
    epsilon = epsilon_start
    best_score = -float('inf')
    
    print("Starting Next-State Board-Evaluation Training...")
    
    for ep in range(episodes):
        env.reset()
        done = False
        step = 0
        ep_reward = 0
        
        while not done:
            next_steps = env.get_next_states()
            if not next_steps: 
                env.step(0)
                break
                
            keys = list(next_steps.keys())
            states = [next_steps[k] for k in keys]
            states_tensor = torch.FloatTensor(states)
            
            # Epsilon-greedy
            if random.random() < epsilon:
                action_idx = random.randint(0, len(keys) - 1)
            else:
                model.eval()
                with torch.no_grad():
                    q_vals = model(states_tensor)
                action_idx = torch.argmax(q_vals).item()
                model.train()
                
            best_action = keys[action_idx]
            chosen_state = states[action_idx]
            
            outcome = env.step_next_state(best_action[0], best_action[1])
            reward = outcome.reward
            done = outcome.done
            ep_reward += reward
            
            memory.append((chosen_state, reward, done))
            
            # Train
            # But wait, in Next-State TD-learning:
            # Q(s) = r + gamma * max_a' Q(s')  <- Standard
            # Because we selected the next state immediately, the memory just stores the state we jumped into!
            # Wait, no. We jump into s', and next turn we evaluate s''.
            # Standard way: we save (state, reward, next_states_batch, done).
            
            # Wait, the simplest way for Next State:
            # Replay buffer stores (state, reward, next_state, done)?
            # Actually, standard next-state uses the features of the result grid.
            # So memory stores (state, reward, next_state, done) where 'state' is the one we CHOSE, 
            # and 'next_state' is the one we CHOOSE in the next step.
            
            step += 1

        epsilon = max(epsilon_end, epsilon - (epsilon_start - epsilon_end)/epsilon_decay)
        
        print(f"Episode {ep}, Reward: {ep_reward:.1f}, Steps: {step}, Epsilon: {epsilon:.3f}")
