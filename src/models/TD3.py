# class StepWrapper:
#     def __init__(self, env, step_hours=3):
#         self.env = env
#         self.step_hours = step_hours
#         self.action_history = []
#         self.reward_log = []

#     def reset(self):
#         return self.env.reset()

#     def step(self, action):
#         total_reward = 0
#         done = False
#         info = {}
#         for _ in range(self.step_hours):
#             if done:
#                 break

#             action -= 1  # Convert action from {0,1,2} to {-1,0,1}
#             obs, reward, terminated, truncated, info = self.env.step(action)
#             self.action_history.append(action)
#             reward = reward_shaping(self.env, reward, self.action_history)
#             total_reward += reward
#             done = terminated or truncated
#         return obs, total_reward, done, done, info
#     def __getattr__(self, name):
#         # alles wat StepWrapper zelf niet heeft, vraag door aan de originele env
#         return getattr(self.env, name)

import gymnasium as gym
import pandas as pd
import numpy as np

class HydroElectricTestEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, path_to_test_data):
        super().__init__()

        # Load data
        self.test_data = pd.read_excel(path_to_test_data)
        self.price_values = self.test_data.iloc[:, 1:25].to_numpy()
        self.timestamps = self.test_data['PRICES']

        self.n_days = self.price_values.shape[0]
        self.max_steps = self.n_days * 24

        # Action space: 0 = sell, 1 = idle, 2 = pump
        self.action_space = gym.spaces.Discrete(3)

        # Observation space
        # [volume, price, hour, day_of_week, day_of_year, month]
        low = np.array([0, 0, 1, 0, 1, 1], dtype=np.float32)
        high = np.array([100000, 1000, 24, 6, 366, 12], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # Physical parameters
        self.max_volume = 100000
        self.max_flow = 18000
        self.volume = self.max_volume / 2

        self.pump_efficiency = 0.8
        self.flow_efficiency = 0.9

        self.water_mass = 1000
        self.dam_height = 30
        self.gravity = 9.81

        self.volume_to_MWh = (
            self.water_mass * self.gravity * self.dam_height * 2.77778e-10
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.day = 0
        self.hour = 0
        self.volume = self.max_volume / 2

        return self._get_obs(), {}

    def step(self, action):
        # Map discrete actions
        action_map = {0: -1, 1: 0, 2: 1}
        action = action_map[int(action)]

        price = self.price_values[self.day, self.hour]
        reward = 0.0

        # Pumping
        if action > 0 and self.volume < self.max_volume:
            flow = min(self.max_flow, self.max_volume - self.volume)
            cost = (flow * self.volume_to_MWh * price) / self.pump_efficiency
            self.volume += flow
            reward = -cost

        # Selling
        elif action < 0 and self.volume > 0:
            flow = min(self.max_flow, self.volume)
            revenue = flow * self.volume_to_MWh * price * self.flow_efficiency
            self.volume -= flow
            reward = revenue

        # Advance time
        self.hour += 1
        if self.hour == 24:
            self.hour = 0
            self.day += 1

        self.step_count += 1

        terminated = self.step_count >= self.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        ts = self.timestamps[self.day]
        return np.array([
            self.volume,
            self.price_values[self.day, self.hour],
            self.hour + 1,
            ts.dayofweek,
            ts.dayofyear,
            ts.month
        ], dtype=np.float32)