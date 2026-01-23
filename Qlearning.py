import numpy as np
import pandas as pd
from reward_shaping import reward_shaping
import matplotlib.pyplot as plt
import optuna 

class QAgent():

    def __init__(self, env, discount_rate=0.95):

        self.env = env
        self.discount_rate = discount_rate
        self.discrete_action_space = self.env.discrete_action_space.n
        self.learning_curve = []
        self.state_visits = None
        
        # Bins 
        self.volume_bins = np.array([-np.inf, 20_000, 40_000, 60_000, 80_000, np.inf])
        self.price_bins = np.array([-np.inf,27,37, 50, 72, np.inf]) #= np.quantile(env.price_values, np.linspace(0, 1, 5 + 1))
        self.hour_bins = np.arange(0, 25, 3)
        self.week_bins = np.array([-np.inf,5 ,np.inf])
        self.month_bins = np.array([-np.inf,3,6,9 ,np.inf])

        # self.bins = [self.volume_bins, self.price_bins, self.hour_bins, self.week_bins, self.month_bins]

        self.action_history = []
        self.total_reward_history = []
        self.val_curve = []

    def discretize_state(self, state):

        volume, price, hour, day_of_week, month = state[0], state[1], state[2], state[3], state[5]

        # volume bins
        v_bin = np.digitize(volume, self.volume_bins) -1
        v_bin = np.clip(v_bin, 0, self.n_volume - 1)

        # price bins
        p_bin = np.digitize(price, self.price_bins)-1
        p_bin = np.clip(p_bin, 0, self.n_price - 1)

        # hour bin 
        h_bin = np.digitize(hour, self.hour_bins)-1
        h_bin = np.clip(h_bin, 0, self.n_hour - 1)

        # week bin 
        w_bin = np.digitize(day_of_week, self.week_bins)-1
        w_bin = np.clip(w_bin, 0, self.n_week - 1)

        # month/seasonal bin 
        m_bin = np.digitize(month, self.month_bins)-1
        m_bin = np.clip(m_bin, 0, self.n_month - 1)

        return (v_bin, p_bin, h_bin,w_bin,m_bin)

    def create_Q_table(self):
        self.n_volume = len(self.volume_bins) - 1
        self.n_price  = len(self.price_bins) - 1
        self.n_hour   = len(self.hour_bins) - 1
        self.n_week = len(self.week_bins)-1
        self.n_month = len(self.month_bins)-1

        self.Qtable = np.zeros((self.n_volume,self.n_price, self.n_hour, self.n_week, self.n_month, 
                                self.discrete_action_space),dtype=np.float32)
        
        self.state_visits = np.zeros((self.n_volume, self.n_price, self.n_hour,self.n_week, self.n_month), 
                                dtype=np.int32)


    def train(self, simulations, learning_rate, epsilon=0.1, epsilon_decay=1000,
              adaptive_epsilon=False, adaptive_learning_rate = True):

        self.create_Q_table()
        self.starting_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.train_curve = []

        self.epsilon_start = 1
        self.epsilon_end = 0.05
        self.epsilon = epsilon

        for episode in range(simulations):

            state, _ = self.env.reset()
            state = self.discretize_state(state)
            
            done = False

            total_rewards = 0
            self.state_rewards =[]
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

                v, p, h,w,m = state
                nv, np_, nh,nw,nm = next_state

                self.state_visits[v, p, h,w,m] += 1

                Q_target = reward + self.discount_rate * np.max(self.Qtable[nv, np_, nh,nw,nm])

                delta = self.learning_rate * (Q_target - self.Qtable[v, p, h,w,m, action])

                self.Qtable[v, p, h,w,m, action] = self.Qtable[v, p, h,w,m, action] + delta

                state = next_state
                total_rewards += reward
                self.state_rewards.append(reward)

            self.train_curve.append(total_rewards)

            if adaptive_learning_rate:
                self.learning_rate *= 0.99
                
            self.total_reward_history.append(total_rewards)

            if episode % 20 == 0:
                print(episode,total_rewards)
                if episode != 0:
                    print("mean reward", round(np.mean(self.state_rewards),2),
                          "min reward", round(min(self.state_rewards),2),
                          "max reward", round(max(self.state_rewards),2), 
                          "max total reward", round(max(self.total_reward_history),2))

            # if adapting_learning_rate:
            #     self.learning_rate = self.learning_rate/np.sqrt(i+1)

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
    
    def hyperparameter_tuning(self, parameter_dict, simulations = 25):

        def objective(trial):
            learning_rate = trial.suggest_float("learning_rate", parameter_dict["learning_rate"][0], parameter_dict["learning_rate"][1], log=True)
            epsilon = trial.suggest_float("epsilon", parameter_dict["epsilon"][0], parameter_dict["epsilon"][1], log=True)

            self.train(simulations=simulations,
                       learning_rate=learning_rate,
                       epsilon=epsilon)

            water_levels, rewards, actions_play = self.play()
            total_reward = sum(rewards)
            return total_reward

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
        print("Best hyperparameters: ", study.best_params)
        self.study = study
        return study.best_params

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



