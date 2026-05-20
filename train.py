import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from environment.foodbank_env import FoodBankEnv
from agents.dqn_agent import DQNAgent
from agents.replay_buffer import ReplayBuffer

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
Path('models').mkdir(exist_ok=True); Path('results').mkdir(exist_ok=True)

def main(episodes=120):
    env = FoodBankEnv(seed=SEED)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    buffer = ReplayBuffer(50000)
    batch_size = 64
    warmup = 300
    epsilon_start, epsilon_end = 1.0, 0.03
    rows = []
    for ep in range(episodes):
        state, _ = env.reset(seed=SEED + ep)
        done = False; total_reward = 0.0; losses = []
        epsilon = max(epsilon_end, epsilon_start * (0.993 ** ep))
        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state; total_reward += reward
            if len(buffer) >= max(batch_size, warmup):
                losses.append(agent.update(buffer.sample(batch_size)))
                agent.sync_target()
        rows.append({'episode': ep+1, 'reward': total_reward, 'loss': float(np.mean(losses)) if losses else 0.0, 'nutrition': info['nutrition'], 'expired_units': info['expired_units'], 'fairness_gap': info['fairness_gap'], 'mean_coverage': info['mean_coverage'], 'mismatches': info['mismatches'], 'epsilon': epsilon})
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1:04d} | reward={total_reward:8.2f} | nutrition={info['nutrition']:.1f} | coverage={info['mean_coverage']:.3f} | eps={epsilon:.3f}")
    agent.save('models/dqn_foodbank.pt')
    df = pd.DataFrame(rows); df.to_csv('results/training_metrics.csv', index=False)
    plt.figure(figsize=(8,4.5)); plt.plot(df['episode'], df['reward'].rolling(20, min_periods=1).mean())
    plt.title('DQN Training Reward (20-episode moving average)'); plt.xlabel('Episode'); plt.ylabel('Positive total reward'); plt.tight_layout(); plt.savefig('results/training_rewards.png', dpi=180)
    print('Saved model to models/dqn_foodbank.pt and metrics to results/training_metrics.csv')

if __name__ == '__main__': main()
