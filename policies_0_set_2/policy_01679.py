def policy(env):
    """
    Maximizes reward by maintaining high speed (always accelerating) and steering to avoid obstacles.
    Scans a region ahead of the player (x=150-200) for cyan obstacles (RGB: 0,255,255) and steers 
    away from the denser cluster (up if obstacles below, down if above). Avoids braking to prevent 
    negative reward and maintain progress. Ties result in no steering change.
    """
    obs = env._get_observation()
    upper_count = 0
    lower_count = 0
    for y in range(100, 300):
        for x in range(150, 200):
            r, g, b = obs[y, x, 0], obs[y, x, 1], obs[y, x, 2]
            if r < 50 and g > 200 and b > 200:
                if y < 200:
                    upper_count += 1
                else:
                    lower_count += 1
    if upper_count > lower_count:
        movement = 2
    elif lower_count > upper_count:
        movement = 1
    else:
        movement = 0
    return [movement, 1, 0]