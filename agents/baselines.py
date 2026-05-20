import numpy as np
class RandomAgent:
    def __init__(self, action_dim): self.action_dim = action_dim
    def act(self, state, epsilon=0.0): return int(np.random.randint(self.action_dim))
class FCFSAgent:
    def __init__(self, n_families=None): pass
    def act(self, state, epsilon=0.0): return 4
class GreedyExpiryAgent:
    def __init__(self, n_families=None): pass
    def act(self, state, epsilon=0.0): return 2
class NutritionFirstAgent:
    def __init__(self, n_families=None): pass
    def act(self, state, epsilon=0.0): return 1
class FairnessFirstAgent:
    def __init__(self, n_families=None): pass
    def act(self, state, epsilon=0.0): return 3
