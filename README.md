# Food Bank Resource Allocation Optimizer

A reinforcement learning decision-support prototype for food-bank donation allocation. The system learns how to allocate food under nutrition, expiry, dietary restriction, budget and fairness constraints.

## Deliverable coverage
- Browser-based web interface: `app.py`
- RL training script: `train.py`
- Saved model: `models/dqn_foodbank.pt`
- Baseline implementations: `agents/baselines.py`
- Environment/simulation: `environment/foodbank_env.py`
- Reproducible test results: `test.py` and `results/evaluation_summary.csv`

## Setup
```bash
pip install -r requirements.txt
```

## Train the RL model
```bash
python train.py
```

## Test against non-RL baselines
```bash
python test.py
```

## Run the web interface
```bash
python -m streamlit run app.py
```

## RL formulation
State includes current food item features, budget, day progress, aggregate coverage/fairness metrics, and recipient need summaries.

Actions represent allocation strategies: balanced RL priority, nutrition-first, expiry-first, fairness-first, first-come-first-served, or skip.

Reward is positive and rewards nutrition delivery, coverage improvement, dietary compatibility, fairness improvement, serving neglected families, and rescuing near-expiry food. It penalizes mismatches, waste, skipping useful allocations, and over-serving.

## Non-RL baselines
- First-Come-First-Served baseline
- Greedy Expiry baseline
- Nutrition-First baseline
- Fairness-First baseline
- Random baseline

## Repository Structure
```text
FoodBank_RL_Optimizer/
├── app.py
├── train.py
├── test.py
├── README.md
├── requirements.txt
├── agents/
│   ├── baselines.py
│   ├── dqn_agent.py
│   └── replay_buffer.py
├── environment/
│   └── foodbank_env.py
├── models/
│   └── dqn_foodbank.pt
├── results/
│   ├── evaluation_summary.csv
│   ├── training_metrics.csv
│   └── *.png
└── docs/
    ├── FoodBank_RL_Project_Report.pdf
    ├── FoodBank_RL_Presentation.pptx
    ├── AI_USAGE_DISCLOSURE.md
    └── INDIVIDUAL_REFLECTIONS.md
```

## Live Demo Steps
1. Run `python -m streamlit run app.py`.
2. Show the starting inventory and recipient snapshots.
3. Click **Run one full comparison episode**.
4. Explain reward, nutrition, expired waste, fairness gap, mean coverage, and mismatches.
5. Mention that the final report results are averaged using `test.py`.

## Notes
The environment uses a fixed seed for reproducible testing. The included model file allows the evaluator to run the dashboard and test script without waiting for retraining.
