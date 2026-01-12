import random
def hourly_policy(observation):
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    if (observation[2] > 8) and (observation[2] < 23):  # If hour is between 9 AM and 10 PM, sell
        action = -1 # Sell 

    else:
        action = 1 # Pump 

    return action

def weekday_policy(observation):
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    if (observation[3] >= 0) and (observation[3] <= 4):  # If day is Monday to Friday, sell
        action = -1 # Sell
    else:
        action = 1 # Pump

    return action

def month_policy(observation):
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    if (observation[5] >= 5) and (observation[5] <= 10):  # If month is May to October, sell
        action = -1 # Sell
    else:
        action = 1 # Pump

    return action

def time_policy(observation):
    # The observation is the tuple: [volume, price, hour_of_day, day_of_week, day_of_year, month_of_year, year]
    # We combine both policies: we only sell when both policies agree to sell and only pump when both policies agree to pump, otherwise we do nothing.
    temp_action1 = hourly_policy(observation) 
    temp_action2 = weekday_policy(observation) 

    if temp_action1 == temp_action2:
        action = temp_action1

    else:
        action = 0 # No action

    return action

def epsilon_greedy_policy(observation, epsilon = 0.1):
    p = random.random()
    if p < epsilon:
        action = random.choice([-1, 0, 1]) # Random action: Sell, No action, or Pump
    else:
        action = time_policy(observation) # Follow the time-based policy
    return action

