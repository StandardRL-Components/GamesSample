def policy(env):
    # Strategy: Direct the snake towards food using Manhattan distance, avoiding walls and body.
    # Prioritize moves that reduce distance to food and are safe (within grid and not in body).
    # Break ties by preferring current direction to minimize oscillation.
    head = env.snake_body[-1]
    food = env.food_pos
    current_dir = env.snake_direction
    actions = []
    
    # Evaluate candidate moves (excluding 180-degree turns)
    for a0, vec in [(1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))]:
        if (vec[0] != -current_dir[0] or vec[1] != -current_dir[1]):  # Prevent reverse
            new_head = (head[0] + vec[0], head[1] + vec[1])
            safe = (0 <= new_head[0] < env.GRID_WIDTH and 
                    0 <= new_head[1] < env.GRID_HEIGHT and 
                    new_head not in env.snake_body)
            dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
            actions.append((a0, safe, dist, vec == current_dir))
    
    # Include no-op (continue current direction)
    new_head = (head[0] + current_dir[0], head[1] + current_dir[1])
    safe = (0 <= new_head[0] < env.GRID_WIDTH and 
            0 <= new_head[1] < env.GRID_HEIGHT and 
            new_head not in env.snake_body)
    dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
    actions.append((0, safe, dist, True))
    
    # Select best action: safe moves with min distance, prefer current direction
    safe_actions = [a for a in actions if a[1]]
    if safe_actions:
        best_dist = min(a[2] for a in safe_actions)
        best_actions = [a for a in safe_actions if a[2] == best_dist]
        # Prefer current direction to reduce oscillation
        best_action = next((a for a in best_actions if a[3]), best_actions[0])
        return [best_action[0], 0, 0]
    
    # If no safe move, choose least unsafe (min distance)
    best = min(actions, key=lambda x: x[2])
    return [best[0], 0, 0]