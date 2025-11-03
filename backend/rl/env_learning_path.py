import gymnasium as gym
import numpy as np
from gymnasium import spaces

class LearningPathEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, topics, activity_types, time_budget=60, candidate_window=None, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.topics = topics
        self.activity_types = activity_types
        self.n_topics = len(topics)
        self.n_acts = len(activity_types)
        self.candidate_window = candidate_window or list(range(self.n_topics))
        self.n_actions = len(self.candidate_window) * self.n_acts
        self.obs_dim = self.n_topics * 3 + 3
        low = np.zeros(self.obs_dim, dtype=np.float32)
        high = np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)
        self.time_budget_init = float(time_budget)
        self.reset_state()

    def reset_state(self):
        self.mastery = self.rng.uniform(0.1, 0.4, size=self.n_topics)
        self.difficulty = self.rng.uniform(0.3, 0.8, size=self.n_topics)
        self.last_seen = np.zeros(self.n_topics, dtype=np.float32)
        self.time_left = float(self.time_budget_init)
        self.global_progress = 0.0
        self.fatigue = 0.0

    def _make_obs(self):
        return np.concatenate([
            self.mastery.astype(np.float32),
            self.difficulty.astype(np.float32),
            self.last_seen.astype(np.float32),
            np.array([
                self.time_left / self.time_budget_init if self.time_budget_init > 0 else 0.0,
                self.global_progress,
                self.fatigue
            ], dtype=np.float32)
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_state()
        return self._make_obs(), {}

    def step(self, action):
        cand_len = len(self.candidate_window)
        topic_pos_in_window = int(action) // self.n_acts
        act_idx = int(action) % self.n_acts
        topic_idx = self.candidate_window[topic_pos_in_window]
        base_time = {0: 8.0, 1: 10.0, 2: 12.0, 3: 15.0}.get(act_idx, 10.0)
        noise = self.rng.normal(0.0, 1.0)
        duration = max(3.0, base_time + noise)
        self.time_left -= duration
        act_gain = {0: 0.02, 1: 0.025, 2: 0.05, 3: 0.06}[act_idx]
        gain = act_gain * (1.0 - self.mastery[topic_idx]) * (0.5 + 0.5 * self.rng.random())
        gain *= (1.0 - 0.3 * self.fatigue)
        pre = float(self.mastery[topic_idx])
        self.mastery[topic_idx] = float(np.clip(self.mastery[topic_idx] + gain, 0.0, 1.0))
        mastery_gain = float(self.mastery[topic_idx] - pre)
        self.last_seen *= 0.95
        self.last_seen[topic_idx] = 1.0
        self.global_progress = float(np.mean(self.mastery))
        self.fatigue = float(np.clip(self.fatigue + duration / 120.0, 0.0, 1.0))
        coverage_bonus = 0.1 if pre < 0.6 and self.last_seen[topic_idx] == 1.0 else 0.0
        time_penalty = 0.2 if self.time_left < 0 else 0.0
        long_step_penalty = 0.02 if duration > 14.0 else 0.0
        reward = 1.0 * mastery_gain + coverage_bonus - time_penalty - long_step_penalty
        terminated = (self.time_left <= 0.0) or (self.global_progress >= 0.95)
        truncated = False
        info = {
            "topic_idx": topic_idx,
            "act_idx": act_idx,
            "duration": duration,
            "mastery_gain": mastery_gain
        }
        return self._make_obs(), float(reward), terminated, truncated, info
