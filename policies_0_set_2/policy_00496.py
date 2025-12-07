def policy(env):
    # Strategy: Maximize speed by always using available boosts to avoid penalty and progress faster.
    # Avoid obstacles by moving to the safest lane (least obstructed) within a 500px lookahead.
    # Collect boosts when safe and aligned, prioritizing obstacle avoidance to prevent collisions (-10 reward).
    # Break ties by maintaining current vertical position to minimize unnecessary movement.

    # Always use boost if available and not active to avoid penalty and maximize horizontal speed
    use_boost = 1 if env.boost_charges > 0 and env.boost_active_timer <= 0 else 0
    
    # Calculate effective horizontal speed for obstacle prediction
    effective_h_speed = env.SNAIL_BASE_H_SPEED
    if env.collision_slowdown_timer > 0:
        effective_h_speed /= 2
    if use_boost and env.boost_charges > 0 and env.boost_active_timer <= 0:
        effective_h_speed = env.SNAIL_BOOST_H_SPEED

    track_top = env.TRACK_Y_POS - env.SNAIL_RADIUS
    track_bottom = env.TRACK_Y_POS + env.TRACK_THICKNESS - env.SNAIL_RADIUS
    lookahead = 500  # pixels ahead to consider for obstacles and boosts

    # Find all obstacles and boosts within lookahead range
    obstacles = []
    boosts = []
    for o in env.obstacles:
        dist = o['x'] - env.world_x
        if 0 < dist < lookahead:
            obstacles.append(o)
    for b in env.boost_items:
        dist = b['x'] - env.world_x
        if 0 < dist < lookahead:
            boosts.append(b)

    # If no obstacles or boosts, maintain current position
    if not obstacles and not boosts:
        return [0, use_boost, 0]

    # Calculate danger zones for obstacles
    danger_zones = []
    for o in obstacles:
        rel_speed = effective_h_speed - o['vx']  # Relative speed between snail and obstacle
        time_to_collision = (o['x'] - env.world_x - 100) / rel_speed  # Time until collision at snail's x=100
        danger_zones.append({
            'y_range': (o['y'], o['y'] + env.OBSTACLE_HEIGHT),
            'time': time_to_collision
        })

    # Evaluate vertical movement options
    best_action = 0  # none
    best_score = -float('inf')
    current_y = env.snail_y

    for test_action in [0, 1, 2]:  # none, up, down
        if test_action == 1 and current_y - env.SNAIL_V_SPEED < track_top:
            continue  # Cannot move up
        if test_action == 2 and current_y + env.SNAIL_V_SPEED > track_bottom:
            continue  # Cannot move down

        # Calculate test y position after movement
        test_y = current_y
        if test_action == 1:
            test_y -= env.SNAIL_V_SPEED
        elif test_action == 2:
            test_y += env.SNAIL_V_SPEED

        # Score based on obstacle avoidance and boost collection
        score = 0
        snail_hitbox = (test_y - env.SNAIL_RADIUS, test_y + env.SNAIL_RADIUS)

        # Penalize collisions with obstacles
        for danger in danger_zones:
            if (snail_hitbox[0] < danger['y_range'][1] and 
                snail_hitbox[1] > danger['y_range'][0]):
                score -= 1000 / max(0.1, danger['time'])  # Higher penalty for imminent collisions

        # Reward boost collection
        for boost in boosts:
            boost_dist = boost['x'] - env.world_x
            if abs(test_y - boost['y']) < env.SNAIL_RADIUS + env.BOOST_RADIUS:
                score += 50 / max(1, boost_dist)  # Higher reward for closer boosts

        # Prefer current position to minimize unnecessary movement
        if test_action == 0:
            score += 5

        if score > best_score:
            best_score = score
            best_action = test_action

    return [best_action, use_boost, 0]