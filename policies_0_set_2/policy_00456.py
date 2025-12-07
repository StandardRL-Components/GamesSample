def policy(env):
    # Strategy: Prioritize moving towards food via shortest Manhattan path while avoiding collisions.
    # If no safe direct path exists, choose the safest available move to avoid game over.
    if env.game_over:
        return [0, 0, 0]
    
    head = env.snake_body[0]
    food = env.food_pos
    current_dir = env.direction
    action_dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    opposite_dir = (-current_dir[0], -current_dir[1])
    
    safe_actions = []
    for action, dir_vec in action_dirs.items():
        if dir_vec == opposite_dir:
            continue
        new_head = (head[0] + dir_vec[0], head[1] + dir_vec[1])
        if not (0 <= new_head[0] < env.GRID_WIDTH and 0 <= new_head[1] < env.GRID_HEIGHT):
            continue
        if new_head in env.snake_body[1:]:
            continue
        safe_actions.append(action)
    
    if safe_actions:
        best_action = min(safe_actions, key=lambda a: abs(head[0] + action_dirs[a][0] - food[0]) + abs(head[1] + action_dirs[a][1] - food[1]))
        return [best_action, 0, 0]
    
    fallback_actions = [a for a in action_dirs if action_dirs[a] != opposite_dir]
    return [fallback_actions[0] if fallback_actions else 0, 0, 0]