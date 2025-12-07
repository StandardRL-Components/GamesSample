def policy(env):
    # Strategy: Use state variables to avoid error-prone pixel detection. Greedily move towards food while avoiding collisions.
    # Prioritize moves that reduce Manhattan distance to food, breaking ties consistently (up, down, left, right).
    # Only consider safe moves (within bounds and not colliding with body except tail). Use current direction as fallback.
    
    head = env.worm_body[0]
    food = env.food_pos
    current_dir = env.direction
    dangerous_body = set(env.worm_body[:-1])  # All body segments except tail
    
    moves = [
        (0, -1, 1),  # up
        (0, 1, 2),   # down
        (-1, 0, 3),  # left
        (1, 0, 4)    # right
    ]
    
    # Filter out opposite direction to prevent reversal
    opposite = (-current_dir[0], -current_dir[1])
    safe_moves = []
    for dx, dy, code in moves:
        if (dx, dy) == opposite:
            continue
        new_pos = (head[0] + dx, head[1] + dy)
        # Check bounds and collision with dangerous body parts
        if (0 <= new_pos[0] < env.GRID_WIDTH and 
            0 <= new_pos[1] < env.GRID_HEIGHT and 
            new_pos not in dangerous_body):
            dist = abs(new_pos[0] - food[0]) + abs(new_pos[1] - food[1])
            safe_moves.append((dist, code, new_pos))
    
    if safe_moves:
        # Choose move that minimizes distance to food
        best_move = min(safe_moves, key=lambda x: x[0])
        return [best_move[1], 0, 0]
    
    # If no safe move found, try current direction if safe
    new_pos = (head[0] + current_dir[0], head[1] + current_dir[1])
    if (0 <= new_pos[0] < env.GRID_WIDTH and 
        0 <= new_pos[1] < env.GRID_HEIGHT and 
        new_pos not in dangerous_body):
        return [0, 0, 0]  # No-op continues current direction
    
    return [0, 0, 0]  # Fallback no-op