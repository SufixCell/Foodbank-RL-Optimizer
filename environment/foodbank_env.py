import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    class _Env: pass
    class _Discrete:
        def __init__(self, n): self.n = n
    class _Box:
        def __init__(self, low, high, shape, dtype): self.low=low; self.high=high; self.shape=shape; self.dtype=dtype
    class _Spaces:
        Discrete = _Discrete; Box = _Box
    class _Gym: Env = _Env
    gym = _Gym(); spaces = _Spaces()

class FoodBankEnv(gym.Env):
    """Food-bank allocation environment.

    Action space is intentionally decision-oriented rather than raw family IDs.
    The RL agent chooses the allocation priority for the current cycle:
      0 = balanced RL priority (need + fairness + dietary fit + expiry)
      1 = nutrition-first priority
      2 = expiry-first priority
      3 = fairness-first priority
      4 = first-come-first-served priority
      5 = skip allocation

    This makes the problem learnable and browser-demo friendly while still
    modeling the real food-bank decision: how should the next donation be
    allocated under scarcity, expiry, nutrition and fairness constraints?
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, n_families=20, n_items=30, episode_length=30, seed=42):
        super().__init__()
        self.n_families = n_families
        self.n_items = n_items
        self.episode_length = episode_length
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(6)
        # item(4) + global(6) + aggregate family stats(8) + top needy family(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(23,), dtype=np.float32)
        self.reset(seed=seed)

    def _generate_inventory(self):
        return {
            "quantity": self.rng.integers(1, 8, size=self.n_items).astype(float),
            "days": self.rng.integers(1, 10, size=self.n_items).astype(float),
            "nutrition": self.rng.integers(35, 100, size=self.n_items).astype(float),
            "category": self.rng.integers(0, 4, size=self.n_items).astype(float),
        }

    def _generate_families(self):
        return {
            "size": self.rng.integers(1, 8, size=self.n_families).astype(float),
            "restriction": self.rng.integers(0, 4, size=self.n_families).astype(float),
            "need": self.rng.uniform(70, 150, size=self.n_families).astype(float),
            "received": np.zeros(self.n_families, dtype=float),
            "last_served": np.ones(self.n_families, dtype=float) * 10,
            "allocations": np.zeros(self.n_families, dtype=float),
        }

    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
        except Exception:
            pass
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.default_rng(seed)
        self.day = 0
        self.budget = 100.0
        self.inventory = self._generate_inventory()
        self.families = self._generate_families()
        self.current_item = 0
        self.total_nutrition = 0.0
        self.expired_units = 0.0
        self.mismatches = 0
        self.history = []
        self.queue_pointer = 0
        return self._get_obs(), {}

    def _coverage(self):
        return self.families["received"] / (self.families["need"] * self.families["size"] + 1e-6)

    def _fairness_gap(self):
        cov = self._coverage()
        return float(np.max(cov) - np.min(cov))

    def _current_item(self):
        return self.current_item % self.n_items

    def _expiry_urgency(self, i):
        return max(0.0, (5.0 - self.inventory["days"][i]) / 5.0)

    def _valid_match(self, i):
        cat = self.inventory["category"][i]
        restriction = self.families["restriction"]
        return (restriction == 0) | (restriction == cat)

    def _select_family(self, action, i):
        cov = self._coverage()
        need_norm = self.families["need"] / 150.0
        size_norm = self.families["size"] / 8.0
        unserved = np.minimum(self.families["last_served"] / 10.0, 1.0)
        valid = self._valid_match(i).astype(float)
        urgent = self._expiry_urgency(i)

        if action == 5:
            return None
        if action == 4:  # FCFS queue style
            for _ in range(self.n_families):
                f = self.queue_pointer % self.n_families
                self.queue_pointer += 1
                if self.families["allocations"][f] < 3:
                    return int(f)
            return int(self.queue_pointer % self.n_families)

        if action == 0:  # balanced decision priority
            score = (2.4 * (1.0 - np.minimum(cov, 1.0)) +
                     1.4 * valid +
                     0.8 * unserved +
                     0.6 * need_norm +
                     0.35 * size_norm +
                     0.7 * urgent * valid)
        elif action == 1:  # nutrition-first
            score = 1.5 * need_norm + 0.6 * size_norm + 0.4 * valid
        elif action == 2:  # expiry-first
            score = 2.0 * urgent * valid + 0.7 * (1.0 - np.minimum(cov, 1.0)) + 0.3 * need_norm
        else:  # fairness-first
            score = 2.2 * (1.0 - np.minimum(cov, 1.0)) + 0.5 * unserved + 0.35 * valid
        return int(np.argmax(score))

    def _get_obs(self):
        i = self._current_item()
        cov = self._coverage()
        valid = self._valid_match(i).astype(float)
        top = int(np.argmin(cov))
        item = np.array([
            min(self.inventory["quantity"][i] / 8.0, 1.0),
            min(self.inventory["days"][i] / 10.0, 1.0),
            min(self.inventory["nutrition"][i] / 100.0, 1.0),
            self.inventory["category"][i] / 3.0,
        ], dtype=np.float32)
        global_features = np.array([
            min(self.budget / 100.0, 1.0),
            self.day / self.episode_length,
            min(float(np.mean(cov)), 1.0),
            min(float(np.min(cov)), 1.0),
            min(self._fairness_gap(), 1.0),
            self._expiry_urgency(i),
        ], dtype=np.float32)
        family_agg = np.array([
            min(float(np.mean(self.families["need"])) / 150.0, 1.0),
            min(float(np.mean(self.families["size"])) / 8.0, 1.0),
            min(float(np.mean(self.families["last_served"])) / 10.0, 1.0),
            min(float(np.std(cov)), 1.0),
            float(np.mean(valid)),
            min(float(np.max(self.families["need"])) / 150.0, 1.0),
            min(float(np.max(self.families["last_served"])) / 10.0, 1.0),
            min(float(np.sum(self.inventory["quantity"])) / (self.n_items * 8.0), 1.0),
        ], dtype=np.float32)
        top_features = np.array([
            top / max(1, self.n_families - 1),
            min(self.families["need"][top] / 150.0, 1.0),
            min(self.families["size"][top] / 8.0, 1.0),
            min(self.families["last_served"][top] / 10.0, 1.0),
            float(valid[top]),
        ], dtype=np.float32)
        return np.concatenate([item, global_features, family_agg, top_features]).astype(np.float32)

    def step(self, action):
        action = int(action)
        i = self._current_item()
        reward = 0.25
        served = None
        item_available = self.inventory["quantity"][i] > 0
        coverage_before = self._coverage()
        fairness_before = self._fairness_gap()
        expiry_urgency = self._expiry_urgency(i)

        if action == 5 or not item_available or self.budget <= 0:
            reward -= 0.65 + 0.9 * expiry_urgency if item_available else 0.1
        else:
            f = self._select_family(action, i)
            served = f
            category = self.inventory["category"][i]
            restriction = self.families["restriction"][f]
            match = (restriction == 0) or (category == restriction)
            old_cov = coverage_before[f]
            old_last = self.families["last_served"][f]
            qty_used = 1.0
            delivered = self.inventory["nutrition"][i] * qty_used
            # mild family-size adjustment to value larger households without making them dominate
            delivered *= (0.75 + 0.25 * min(self.families["size"][f] / 4.0, 1.5))

            self.families["received"][f] += delivered
            self.families["last_served"][f] = 0
            self.families["allocations"][f] += 1
            self.inventory["quantity"][i] -= qty_used
            self.budget -= 1.5
            self.total_nutrition += delivered

            coverage_after = self._coverage()
            coverage_gain = max(0.0, coverage_after[f] - old_cov)
            fairness_improvement = max(0.0, fairness_before - self._fairness_gap())
            undercovered = max(0.0, 1.0 - old_cov)
            neglect = min(old_last / 10.0, 1.0)

            reward += delivered / 20.0
            reward += 20.0 * coverage_gain
            reward += 7.0 * fairness_improvement
            reward += 1.25 * undercovered
            reward += 0.70 * neglect
            reward += 1.20 * expiry_urgency
            if match:
                reward += 2.50
            else:
                reward -= 5.00
                self.mismatches += 1
            if old_cov > 0.65:
                reward -= 2.25 * old_cov

        self.inventory["days"] -= 0.12
        expired_mask = self.inventory["days"] <= 0
        newly_expired = float(np.sum(self.inventory["quantity"][expired_mask]))
        if newly_expired > 0:
            reward -= 0.90 * newly_expired
            self.expired_units += newly_expired
            self.inventory["quantity"][expired_mask] = 0
            self.inventory["days"][expired_mask] = 0

        self.families["last_served"] += 1
        self.current_item = (self.current_item + 1) % self.n_items
        self.day += 1
        terminated = self.day >= self.episode_length
        truncated = False

        if terminated:
            mean_cov = float(np.mean(self._coverage()))
            fairness_gap = self._fairness_gap()
            mismatch_rate = self.mismatches / max(1, self.episode_length)
            reward += 22.0 * mean_cov
            reward += 8.0 * max(0.0, 1.0 - fairness_gap)
            reward += 5.0 * max(0.0, 1.0 - mismatch_rate)

        reward = max(0.05, float(reward))
        info = {
            "nutrition": float(self.total_nutrition),
            "expired_units": float(self.expired_units),
            "fairness_gap": float(self._fairness_gap()),
            "mean_coverage": float(np.mean(self._coverage())),
            "served_family": served,
            "mismatches": int(self.mismatches),
        }
        self.history.append(info.copy())
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"Day {self.day} | nutrition={self.total_nutrition:.1f} | waste={self.expired_units:.1f} | fairness_gap={self._fairness_gap():.3f}")
