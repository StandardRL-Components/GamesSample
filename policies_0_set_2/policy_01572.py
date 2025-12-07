def policy(env):
    # Strategy: Move bucket to catch the lowest (most urgent) critter to maximize catches and minimize escapes.
    # Prioritize critters closest to the bottom since they must be caught immediately to avoid penalties.
    if env.game_over or not env.critters:
        return [0, 0, 0]
    
    lowest_critter = max(env.critters, key=lambda c: c['pos'][1])
    bucket_center = env.bucket_x + env.bucket_width / 2
    target_x = lowest_critter['pos'][0]
    
    if abs(bucket_center - target_x) < 5:
        return [0, 0, 0]
    elif bucket_center > target_x:
        return [3, 0, 0]
    else:
        return [4, 0, 0]