def policy(env):
    # Strategy: Navigate toward exit by selecting wormhole in direction that minimizes distance to exit while avoiding obstacles.
    # Prioritize moves that reduce distance to exit, with safety checks to avoid collisions.
    current_pos = env.wormholes[env.current_wormhole_idx]['pos']
    exit_center = env.exit_rect.center
    best_dir = 0
    min_dist = float('inf')
    
    for direction in [1, 2, 3, 4]:
        target_idx = env._find_target_wormhole(direction)
        if target_idx is None:
            continue
        target_pos = env.wormholes[target_idx]['pos']
        # Check for obstacle collisions
        safe = True
        for obs in env.obstacles:
            if obs['rect'].collidepoint(target_pos):
                safe = False
                break
        if not safe:
            continue
        dist = target_pos.distance_to(exit_center)
        if dist < min_dist:
            min_dist = dist
            best_dir = direction
    
    return [best_dir, 0, 0]