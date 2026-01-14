def penalty_volume(env):
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    dam_level = env.observation()[0]
    max_volume = 0.9 * env.max_volume
    min_volume = 0.1 * env.max_volume

    # Apply penalty for exceeding max or min volume
    if dam_level > max_volume:
        penalty = -50 * (dam_level - max_volume) / max_volume
    elif dam_level < min_volume:
        penalty = -50 * (min_volume - dam_level) / min_volume
    else:
        penalty = 0

    return penalty

def penalty_overflow(env):
    dam_level = env.observation()[0]
    if dam_level > env.max_volume:
        penalty = -100
    else:
        penalty = 0
    return penalty

def penalty_bursts(action_history):
    penalty = 0

    # Apply penalty for 3 consecutive identical actions
    if abs(sum(action_history[-3:])) == 3:
        penalty = -5

    # Apply penalty for extreme action changes
    current_action = action_history[-1]
    previous_action = action_history[-2]
    if abs(current_action - previous_action) >= 2:
        penalty = -10
    
    return penalty

def reward_shaping(env, base_reward, action_history):
    # Calculate volume penalty
    p_volume = penalty_volume(env)
    p_overflow = penalty_overflow(env)
    p_bursts = penalty_bursts(action_history)

    shaped_reward = base_reward + p_volume + p_overflow + p_bursts
    return shaped_reward

