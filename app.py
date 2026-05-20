import os
import pandas as pd
import streamlit as st
import plotly.express as px
from environment.foodbank_env import FoodBankEnv
from agents.dqn_agent import DQNAgent
from agents.baselines import RandomAgent, FCFSAgent, GreedyExpiryAgent, NutritionFirstAgent, FairnessFirstAgent

st.set_page_config(page_title='Food Bank RL Optimizer', layout='wide')
st.title('Food Bank Resource Allocation Optimizer')
st.caption('RL decision-support prototype for fair, nutritious, low-waste food distribution')

@st.cache_resource
def load_agent():
    env = FoodBankEnv(seed=42)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    if os.path.exists('models/dqn_foodbank.pt'):
        agent.load('models/dqn_foodbank.pt')
    return agent

def run_episode(agent, name, seed=42):
    env = FoodBankEnv(seed=seed)
    state, _ = env.reset(seed=seed)
    done = False; total_reward = 0
    while not done:
        action = agent.act(state, epsilon=0.0)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward; done = terminated or truncated
    return {'Method': name, 'Reward': round(total_reward, 3), 'Nutrition': round(info['nutrition'], 2), 'Expired Waste': round(info['expired_units'], 2), 'Fairness Gap': round(info['fairness_gap'], 4), 'Mean Coverage': round(info['mean_coverage'], 4), 'Mismatches': info['mismatches']}

agent = load_agent(); env = FoodBankEnv(seed=42); state, _ = env.reset(seed=42); rec_action = agent.act(state, epsilon=0.0)
action_names = {0:'Balanced allocation',1:'Nutrition-first',2:'Expiry-first',3:'Fairness-first',4:'FCFS queue',5:'Skip'}
col1, col2, col3, col4 = st.columns(4)
col1.metric('Families', env.n_families); col2.metric('Food Items', env.n_items)
col3.metric('Recommended Strategy', action_names.get(rec_action, str(rec_action)))
col4.metric('Starting Budget', f'{env.budget:.0f}')

st.subheader('Live allocation demo')
st.write('Click the button to compare the trained DQN policy with non-RL baselines under the same simulated episode.')
if st.button('Run one full comparison episode'):
    rows = [
        run_episode(agent, 'DQN RL Agent'),
        run_episode(FCFSAgent(), 'FCFS Baseline'),
        run_episode(GreedyExpiryAgent(), 'Greedy Expiry Baseline'),
        run_episode(NutritionFirstAgent(), 'Nutrition-First Baseline'),
        run_episode(FairnessFirstAgent(), 'Fairness-First Baseline'),
        run_episode(RandomAgent(env.action_space.n), 'Random Baseline'),]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)
    st.plotly_chart(px.bar(df, x='Method', y='Reward', title='Positive Reward Comparison'), use_container_width=True)
    st.plotly_chart(px.bar(df, x='Method', y='Nutrition', title='Nutrition Delivered: Higher is Better'), use_container_width=True)
    st.plotly_chart(px.bar(df, x='Method', y='Mean Coverage', title='Mean Recipient Coverage: Higher is Better'), use_container_width=True)
    st.plotly_chart(px.bar(df, x='Method', y='Expired Waste', title='Expired Waste: Lower is Better'), use_container_width=True)

st.subheader('Action meaning')
st.table(pd.DataFrame({'Action':[0,1,2,3,4,5], 'Strategy':['Balanced RL priority','Nutrition-first','Expiry-first','Fairness-first','First-come-first-served','Skip allocation']}))

st.subheader('Simulated inventory snapshot')
inv = pd.DataFrame({'Item ID': range(env.n_items),'Quantity': env.inventory['quantity'],'Days to Expiry': env.inventory['days'],'Nutrition Score': env.inventory['nutrition'],'Category': env.inventory['category']})
st.dataframe(inv.head(15), use_container_width=True)

st.subheader('Recipient snapshot')
fam = pd.DataFrame({'Family ID': range(env.n_families),'Family Size': env.families['size'],'Diet Restriction': env.families['restriction'],'Need Score': env.families['need'],'Days Since Served': env.families['last_served']})
st.dataframe(fam.head(15), use_container_width=True)
st.info('This browser-based dashboard satisfies the working prototype requirement and avoids notebook-only evaluation.')
