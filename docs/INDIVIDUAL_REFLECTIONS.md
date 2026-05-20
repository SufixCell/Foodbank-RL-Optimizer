# Individual Reflections — CT-469 Project

## AI 22047 — Sufiyan Nadeem

**Specific contribution:** Worked on the RL formulation, environment design review, reward tuning, DQN training/testing, dashboard testing, and final result interpretation. Main related files/components: `environment/foodbank_env.py`, `agents/dqn_agent.py`, `train.py`, `test.py`, and `app.py`.

**Most difficult technical challenge:** The hardest part was designing a reward function that balanced nutrition, fairness, expiry reduction, dietary compatibility, and coverage without making the agent optimize only one objective. This was resolved by using a positive multi-objective reward and evaluating it against several baselines.

**AI tool usage:** Used ChatGPT for starter structure, debugging PyTorch/Streamlit issues, reward design suggestions, report/presentation drafting, and result interpretation. The code and outputs were tested and reviewed locally.

**New RL understanding:** Learned that the MDP design, especially state representation, action design, and reward shaping, can matter as much as the neural network architecture.

## AI 22041 — Talha Noor

**Specific contribution:** Supported baseline planning, evaluation comparison, documentation review, and demo validation. Main related files/components: `agents/baselines.py`, `test.py`, report review, and presentation review.

**Most difficult technical challenge:** The main challenge was comparing RL fairly against non-RL methods under the same environment conditions. This was handled by using the same seeds, metrics, and evaluation episodes for all methods.

**AI tool usage:** Used AI assistance to understand baseline comparison wording, documentation structure, and presentation explanation. Final explanations and review were done by the group.

**New RL understanding:** Learned that a baseline is necessary to prove whether RL is actually useful, because a simple heuristic can sometimes perform close to a trained agent.

## AI 22044 — Ali Rana

**Specific contribution:** Supported web interface checking, result screenshots, experiment verification, and live demo preparation. Main related files/components: `app.py`, `results/*.png`, Streamlit demo testing, and presentation assets.

**Most difficult technical challenge:** The main challenge was making the project demo-friendly and understandable through graphs and a browser interface instead of only scripts. This was handled by using Streamlit to show inventory, recipients, and comparison charts.

**AI tool usage:** Used AI assistance for presentation wording, visual result explanation, and demo preparation. The generated material was checked against the actual results.

**New RL understanding:** Learned that RL project evaluation is not only about training reward; it also requires explaining policies, baselines, fairness trade-offs, and real-world limitations.
