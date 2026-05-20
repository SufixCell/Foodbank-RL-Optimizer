import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from environment.foodbank_env import FoodBankEnv
from agents.dqn_agent import DQNAgent
from agents.baselines import RandomAgent, FCFSAgent, GreedyExpiryAgent, NutritionFirstAgent, FairnessFirstAgent

SEED = 42
Path('results').mkdir(exist_ok=True)

def evaluate(agent, name, episodes=50):
    rows = []
    for ep in range(episodes):
        env = FoodBankEnv(seed=SEED + ep)
        state, _ = env.reset(seed=SEED + ep)
        done = False; total_reward = 0.0
        while not done:
            action = agent.act(state, epsilon=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated; total_reward += reward
        rows.append({'method': name, 'reward': total_reward, **info})
    return rows

def main():
    probe = FoodBankEnv(seed=SEED)
    dqn = DQNAgent(probe.observation_space.shape[0], probe.action_space.n)
    dqn.load('models/dqn_foodbank.pt')
    all_rows = []
    all_rows += evaluate(dqn, 'DQN RL Agent')
    all_rows += evaluate(FCFSAgent(), 'First-Come-First-Served Baseline')
    all_rows += evaluate(GreedyExpiryAgent(), 'Greedy Expiry Baseline')
    all_rows += evaluate(NutritionFirstAgent(), 'Nutrition-First Baseline')
    all_rows += evaluate(FairnessFirstAgent(), 'Fairness-First Baseline')
    all_rows += evaluate(RandomAgent(probe.action_space.n), 'Random Baseline')
    df = pd.DataFrame(all_rows)
    summary = df.groupby('method').agg({'reward':'mean','nutrition':'mean','expired_units':'mean','fairness_gap':'mean','mean_coverage':'mean','mismatches':'mean'}).reset_index()
    order = ['DQN RL Agent','First-Come-First-Served Baseline','Greedy Expiry Baseline','Nutrition-First Baseline','Fairness-First Baseline','Random Baseline']
    summary['method'] = pd.Categorical(summary['method'], order, ordered=True); summary = summary.sort_values('method')
    summary.to_csv('results/evaluation_summary.csv', index=False)
    print(summary)
    for col, title, ylabel in [
        ('reward','Positive Reward: RL vs Non-RL Baselines','Mean positive reward'),
        ('nutrition','Nutrition Delivered Comparison','Mean nutrition delivered'),
        ('mean_coverage','Mean Recipient Coverage Comparison','Mean coverage'),
        ('fairness_gap','Fairness Gap Comparison (Lower is Better)','Fairness gap')]:
        plt.figure(figsize=(10,5)); plt.bar(summary['method'].astype(str), summary[col])
        plt.xticks(rotation=20, ha='right'); plt.ylabel(ylabel); plt.title(title); plt.tight_layout(); plt.savefig(f"results/{col}_comparison.png", dpi=180)
    return summary
if __name__ == '__main__': main()
