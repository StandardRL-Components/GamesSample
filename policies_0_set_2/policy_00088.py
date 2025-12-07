def policy(env):
    # Strategy: Jump when an obstacle is within a calculated distance threshold and the beat timing is optimal.
    # This maximizes reward by avoiding collisions, clearing obstacles with perfect timing bonuses, and minimizing air time penalties.
    if env.is_jumping:
        return [0, 0, 0]
    
    step_mod = env.steps % env.BEAT_PERIOD
    in_perfect_window = (step_mod <= 2) or (step_mod >= 28)
    
    perfect_min = 15 * env.obstacle_speed - 10
    perfect_max = 15 * env.obstacle_speed + 10
    emergency_threshold = 5 * env.obstacle_speed
    
    obstacle_near = False
    emergency = False
    
    for obstacle in env.obstacles:
        if obstacle['cleared']:
            continue
        dist = obstacle['x'] - env.PLAYER_X
        if dist < 0:
            continue
        if perfect_min <= dist <= perfect_max and in_perfect_window:
            obstacle_near = True
            break
        if dist <= emergency_threshold:
            emergency = True
            break
    
    if obstacle_near or emergency:
        return [0, 1, 0]
    
    return [0, 0, 0]