import numpy as np
import pandas as pd
from reward_shaping import reward_shaping

def discretize_state(obs):
    volume, price, hour = obs[0], obs[1], obs[2]

    volume_bin = int(volume // 10000)          # 0–10
    price_bin = min(int(price // 10), 19)      # 0–19
    hour_bin = int(hour - 1)                   # 0–23

    return volume_bin, price_bin, hour_bin

class QAgent():

    def __init__(self, env, discount_rate=0.95,
                 volume_bins=10, price_bins=10):

        self.env = env
        self.discount_rate = discount_rate
        self.discrete_action_space = self.env.discrete_action_space.n
        self.learning_curve = []

        self.state_visits = None

        # --- bins ---
        # self.volume_bins = np.linspace(0, 100000, volume_bins)
        # self.price_bins = np.linspace( 0, np.max(env.price_values), price_bins)
        # self.hour_bins = np.arange(24)
        self.volume_bins = np.array([-np.inf, 20_000, 40_000, 60_000, 70_000, np.inf]) #30_000
        # self.price_bins = np.array([-np.inf, 20, 40, 50, 60, 80, np.inf])
        self.price_bins = np.quantile(env.price_values, np.linspace(0, 1, 5 + 1))
        self.hour_bins = np.arange(0, 24, 1)

        self.bins = [self.volume_bins, self.price_bins, self.hour_bins]

        self.action_history = []
        self.total_reward_history = []

    def discretize_state(self, state):

        volume, price, hour = state[0], state[1], state[2]

        # v_bin = np.digitize(volume, self.volume_bins) - 1
        # p_bin = np.digitize(price, self.price_bins) - 1
        # # h_bin = int(hour - 1)
        # h_bin = np.digitize(hour, self.hour_bins) - 1
        # h_bin = np.clip(h_bin, 0, len(self.hour_bins)-2)


        # v_bin = np.clip(v_bin, 0, len(self.volume_bins) - 2)
        # p_bin = np.clip(p_bin, 0, len(self.price_bins) - 2)

        # volume bins
        v_bin = np.digitize(volume, self.volume_bins) -1

        # price bins
        p_bin = np.digitize(price, self.price_bins)-1

        # hour bin (exact)
        h_bin = np.digitize(price, self.hour_bins)-1


        return (v_bin, p_bin, h_bin)

    def create_Q_table(self):

        self.Qtable = np.zeros((len(self.volume_bins),
                                len(self.price_bins),
                                len(self.hour_bins),
                                self.discrete_action_space),dtype=np.float32)
        self.state_visits = np.zeros((
                    len(self.volume_bins),
                    len(self.price_bins),
                    len(self.hour_bins)), dtype=np.int32)


    def train(self, simulations, learning_rate,
              epsilon=0.1, epsilon_decay=10000,
              adaptive_epsilon=False):

        self.create_Q_table()
        self.learning_rate = learning_rate

        self.epsilon_start = 1
        self.epsilon_end = 0.05
        self.epsilon = epsilon

        for episode in range(simulations):

            state, _ = self.env.reset()
            state = self.discretize_state(state)
            
            done = False

            total_rewards = 0
            self.action_history = []

            if adaptive_epsilon:
                self.epsilon = np.interp(episode,[0, epsilon_decay], [self.epsilon_start, self.epsilon_end])

            while not done:

                if np.random.uniform(0, 1) < self.epsilon:
                    action = self.env.discrete_action_space.sample()
                else:
                    action = np.argmax(self.Qtable[state])

                env_action =  action - 1
                next_state, base_reward, terminated, truncated, _ = self.env.step(env_action)
                reward = reward_shaping(self.env, base_reward, self.action_history)
                self.action_history.append(env_action)
                done = terminated or truncated
                next_state = self.discretize_state(next_state)

                v, p, h = state
                nv, np_, nh = next_state

                self.state_visits[v, p, h] += 1

                Q_target = reward + self.discount_rate * np.max(self.Qtable[nv, np_, nh])

                delta = self.learning_rate * (Q_target - self.Qtable[v, p, h, action])

                self.Qtable[v, p, h, action] = self.Qtable[v, p, h, action] + delta

                state = next_state
                total_rewards += reward

            self.learning_curve.append(total_rewards)
            self.total_reward_history.append(total_rewards)

            if episode % 20 == 0:
                print(episode,total_rewards)
                if episode != 0:
                    print(np.mean(self.total_reward_history[20:]),min(self.total_reward_history[20:]),max(self.total_reward_history[20:]))

            # if adapting_learning_rate:
            #     self.learning_rate = self.learning_rate/np.sqrt(i+1)
           
        # print("Training finished")

    def play(self):

        water_levels = []
        rewards = []
        actions_play = []
        state, _ = self.env.reset()
        done = False
        # i = 0

        while not done:
            # i += 1
            state_d = self.discretize_state(state)
            action = np.argmax(self.Qtable[state_d])
            action -= 1
            state, reward, terminated, truncated, _ = self.env.step(action)
            actions_play.append(action)
            done = terminated or truncated
            water_levels.append(state[0])
            rewards.append(reward)
        
            # if i >= 365:
            #     done = True

        
        # Print totale reward
        print("Total reward in evaluation:", round(sum(rewards),2))
        print("avg reward", round(np.mean(rewards),2))
        print("max", max(rewards), "min",min(rewards))
        return water_levels, rewards, actions_play


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



