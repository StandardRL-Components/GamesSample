def policy(env):
    # Greedy Manhattan distance minimization to next target number, avoiding walls and breaking ties consistently (up, down, left, right).
    target_val = env.next_number_to_collect
    if target_val > env.NUM_TARGETS:
        return [0, 0, 0]  # All numbers collected
    
    target = env._find_number(target_val)
    if target is None:
        return [0, 0, 0]  # Target not found (shouldn't happen)
    
    tx, ty = target['pos']
    px, py = env.player_pos
    best_action = 0
    best_dist = float('inf')
    
    # Check each movement direction for validity and distance reduction
    for action, (dx, dy) in [(1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))]:
        wall_key = {1: 'N', 2: 'S', 3: 'W', 4: 'E'}[action]
        if not env.maze[py][px][wall_key]:
            new_dist = abs(px + dx - tx) + abs(py + dy - ty)
            if new_dist < best_dist:
                best_dist = new_dist
                best_action = action
    
    return [best_action, 0, 0]