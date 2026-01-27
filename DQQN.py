import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box
from reward_shaping import reward_shaping

class DQNWrapper(gym.Env):
    def __init__(self, env, reward_shape, num_actions=3):
        super().__init__()
        self.env = env
        self.max_episode_length = len(self.env.price_values.flatten())
        self.actions = np.linspace(-1.0, 1.0, num_actions)
        self.action_space = Discrete(num_actions)
        self.reward_shape = reward_shape

        # 5 discrete features
        self.observation_space = Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_history = []

    def normalize(self, state): 
        state = state.astype(np.float32) 
        state[0] /= self.env.max_volume 
        state[1] /= np.max(self.env.price_values) 
        state[2] = (state[2]-1) /23.0 
        state[3] /= 6.0 
        state[5] = (state[5]-1)/ 12.0 

        return np.array([state[0], state[1], state[2], state[3], state[5]],dtype=np.float32)


    def reset(self, seed=None, **kwargs):
        self.env.counter = 0
        self.env.hour = 1
        self.env.day = 1
        self.env.volume = self.env.max_volume / 2
        self.action_history = []
        self.env.reset()
        obs = np.array(self.env.observation(), dtype=np.float32)
        return self.normalize(obs), {}
    

    def step(self, action):
        
        real_action = self.actions[int(action)]
        self.action_history.append(real_action)

        next_obs, reward, terminated, truncated, info = self.env.step(real_action)
        shaped_reward = reward

        if self.reward_shape: 
            shaped_reward = reward_shaping(self.env, reward, self.action_history)

        # Episode van 7 dagen (168 stappen)
        # if self.env.counter % (1068) == 0 and self.env.counter > 0:
        #     truncated = True

        next_obs = self.normalize(np.array(next_obs, dtype=np.float32))

        return next_obs, shaped_reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)
