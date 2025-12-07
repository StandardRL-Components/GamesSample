def policy(env):
    """
    Maximizes reward by efficiently mining asteroids while managing fuel and avoiding collisions.
    Prioritizes closest safe asteroids for mining, uses minimal movement to conserve fuel,
    and avoids collisions by maintaining safe distances. Stops thrusting when in mining range.
    """
    player_pos = env.player_pos
    asteroids = env.asteroids
    mining_distance = env.PLAYER_SIZE * env.MINING_DISTANCE_FACTOR
    collision_distance = env.PLAYER_SIZE * 0.8
    
    # Check if currently mining valid target
    if env.is_mining and env.mining_target in asteroids:
        dist = player_pos.distance_to(env.mining_target['pos'])
        if dist < env.mining_target['radius'] + mining_distance:
            return [0, 1, 0]  # Continue mining without movement
    
    # Find best asteroid target (closest safe asteroid)
    best_asteroid = None
    min_dist = float('inf')
    for asteroid in asteroids:
        dist = player_pos.distance_to(asteroid['pos'])
        if dist < asteroid['radius'] + collision_distance:
            continue  # Skip asteroids too close for safety
        if dist < min_dist:
            min_dist = dist
            best_asteroid = asteroid
    
    if best_asteroid is None:
        return [0, 0, 0]  # No safe targets, wait
    
    # Move toward target asteroid if not in mining range
    target_pos = best_asteroid['pos']
    dist_to_target = player_pos.distance_to(target_pos)
    if dist_to_target > best_asteroid['radius'] + mining_distance:
        dx = target_pos.x - player_pos.x
        dy = target_pos.y - player_pos.y
        if abs(dx) > abs(dy):
            movement = 4 if dx > 0 else 3  # Right/Left
        else:
            movement = 2 if dy > 0 else 1  # Down/Up
        return [movement, 0, 0]
    else:
        return [0, 1, 0]  # In range, start mining