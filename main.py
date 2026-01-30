from stable_baselines3 import DQN, PPO
from src.models.Qlearning import QAgent
from src.env.TestEnv import HydroElectric_Test
from src.utils.policies import hourly_policy, weekday_policy, time_policy, epsilon_greedy_policy
from src.models.DQN import DQNWrapper
from src.models.PPO import PPOWrapper
from src.models.Qlearning import number_of_actions_env_wrapper
from src.utils.visualizations import create_performance_dashboard
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--excel_file', type=str, default='data/validate.xlsx') # Path to the excel file with the test data
parser.add_argument('--model', type=str, default='dqn', choices=['dqn', 'ppo', 'qlearning', 'policy'],
                    help='Model to use: dqn, ppo, qlearning, or policy (default: dqn)')
parser.add_argument('--model_path', type=str, default='pretrained_models/dqn_model_5bins_reward_shaping.zip',
                    help='Path to the pretrained model file (default: pretrained_models/dqn_model_5bins_reward_shaping.zip)')
parser.add_argument('--policy_type', type=str, default='time', choices=['hourly', 'weekday', 'time', 'epsilon'],
                    help='Policy type to use when model=policy (default: time)')
args = parser.parse_args()

# Initialize the environment
env = HydroElectric_Test(path_to_test_data=args.excel_file)

total_reward = []
water_level = []
action_history = []



if args.model.lower() == 'dqn':
    print("Loading DQN model from {args.model_path}...".format(args=args))

    # Wrap the environment
    env = DQNWrapper(env, reward_shape= False, num_actions= 5)

    #reset the environment
    observation, _ = env.reset(seed = 5)
    observation = np.array(observation, dtype=np.float32)
    model = DQN.load(args.model_path, env=env)

    done = False

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        action = int(action)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward.append(reward)
        water_level.append(env.env.volume)
        action_history.append(action)
        
        observation = np.array(observation, dtype=np.float32)
    print("Total reward DQN: ", sum(total_reward))
    create_performance_dashboard(water_level, total_reward, action_history, max_volume = 100000)
        
if args.model.lower() == 'ppo':
    args.model_path = 'pretrained_models/ppo_model_5bins_reward_shaping.zip'
    print("Loading PPO model from {args.model_path}...".format(args=args))

    # Wrap the environment
    env = PPOWrapper(env, reward_shape= False)

    #reset the environment
    observation, _ = env.reset(seed = 5)
    observation = np.array(observation, dtype=np.float32)
    model = PPO.load(args.model_path, env=env)

    done = False

    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward.append(reward)
        water_level.append(env.env.volume)
        action_history.append(action)
        
        observation = np.array(observation, dtype=np.float32)
    print("Total reward PPO: ", sum(total_reward))
    create_performance_dashboard(water_level, total_reward, action_history, max_volume = 100000)

if args.model.lower() == 'qlearning':
    args.model_path = 'pretrained_models/q_agent3.npz'
    print("Using Q-learning policy...")

    #reset the environment
    env.reset(seed = 5)
    env = number_of_actions_env_wrapper(env, 5)

    agent = QAgent(env)
    agent.load(args.model_path)
    water_level, reward, action_history = agent.play()
    print("Total reward Q-learning: ", sum(reward))
    create_performance_dashboard(water_level, reward, action_history, max_volume = 100000)

if args.model.lower() == 'policy':
    print("Using predefined policy: {args.policy_type}...".format(args=args))

    #reset the environment
    env.reset(seed = 5)
    observation = env.observation()

    for i in range(len(env.test_data)*24 - 1):
        if args.policy_type.lower() == 'hourly':
            action = hourly_policy(observation)
        elif args.policy_type.lower() == 'weekday':
            action = weekday_policy(observation)
        elif args.policy_type.lower() == 'time':
            action = time_policy(observation)
        elif args.policy_type.lower() == 'epsilon':
            action = env.epsilon_greedy_policy(observation, epsilon=0.1)
        else:
            raise ValueError("Invalid policy type. Choose from 'hourly', 'weekday', or 'time'.")

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward.append(reward)
        water_level.append(observation[0])
        action_history.append(action)
        observation = next_observation

    print("Total reward predefined policy: ", sum(total_reward))
    create_performance_dashboard(water_level, total_reward, action_history, max_volume = 100000)