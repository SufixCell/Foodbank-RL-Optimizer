from environment.foodbank_env import FoodBankEnv
from agents.dqn_agent import DQNAgent
from agents.baselines import RandomAgent, FCFSAgent, GreedyExpiryAgent, NutritionFirstAgent, FairnessFirstAgent
from agents.replay_buffer import ReplayBuffer

if __name__ == "__main__":
    env = FoodBankEnv(seed=42)
    state, _ = env.reset(seed=42)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    action = agent.act(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    print("Project import check passed.")
    print("Observation shape:", env.observation_space.shape)
    print("Action dim:", env.action_space.n)
    print("Sample reward:", round(float(reward), 3))
    print("Info keys:", sorted(info.keys()))
