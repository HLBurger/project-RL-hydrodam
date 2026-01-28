def penalty_overflow(env):
    dam_level = env.observation()[0]
    if dam_level >= env.max_volume:
        penalty = -100
    else:
        penalty = 0
    return penalty

def penalty_underflow(env):
    dam_level = env.observation()[0]
    if dam_level <= 0:
        penalty = -100
    else:
        penalty = 0
    return penalty

# def penalty_bursts(action_history):
#     penalty = 0

#     if len(action_history) < 2:
#         return penalty

#     # Apply penalty for extreme action changes
#     current_action = action_history[-1]
#     previous_action = action_history[-2]
#     if abs(current_action - previous_action) >= 2:
#         penalty = -15
    
#     return penalty

# def penalty_idle(action_history):
#     penalty = 0

#     if len(action_history) < 24:
#         return penalty
    
#     if action_history[-1] != 0:
#         return penalty
    
#     # Apply penalty for each hour of idling in the last 24 hours
#     idle_hours = action_history[-24:].count(0)
#     # penalty = -2 * idle_hours
#     if idle_hours >= 22:
#         penalty = -2 * idle_hours

#     return penalty

def reward_potential(env):
    dam_level = env.observation()[0]
    potential_reward = (dam_level / env.max_volume) * 5
    return potential_reward

def reward_shaping(env, base_reward, action_history):
    # Calculate volume penalty
    p_overflow = penalty_overflow(env)
    p_underflow = penalty_underflow(env)
    # p_bursts = penalty_bursts(action_history)
    # p_idle = penalty_idle(action_history)
    r_potential = reward_potential(env)

    shaped_reward = base_reward + p_overflow + p_underflow + r_potential
    return shaped_reward



