def policy(env):
    # Strategy: Jump only when an obstacle is within critical distance, accounting for speed and width to avoid premature jumps and maximize successful clears.
    if env.is_jumping:
        return [0, 0, 0]
    critical_time = 20  # Adjusted timing based on jump physics and obstacle speed
    for obstacle in env.obstacles:
        if obstacle['cleared']:
            continue
        distance = obstacle['x'] - env.PLAYER_X
        if distance < 0:  # Skip obstacles already passed
            continue
        time_to_cross = distance / env.obstacle_speed
        if time_to_cross <= critical_time:
            return [0, 1, 0]
    return [0, 0, 0]