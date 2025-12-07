def policy(env):
    # Strategy: Prioritize moving towards the apple using Manhattan distance while avoiding collisions.
    # This minimizes distance to reward and prevents termination, maximizing long-term score.
    head = env.snake_body[0]
    apple = env.apple_pos
    current_dir = env.direction
    candidates = []
    
    # Evaluate all possible movement actions (including no-op)
    for a0 in range(5):
        if a0 == 0:
            new_dir = current_dir
        elif a0 == 1 and current_dir != (0, 1):
            new_dir = (0, -1)
        elif a0 == 2 and current_dir != (0, -1):
            new_dir = (0, 1)
        elif a0 == 3 and current_dir != (1, 0):
            new_dir = (-1, 0)
        elif a0 == 4 and current_dir != (-1, 0):
            new_dir = (1, 0)
        else:
            new_dir = current_dir  # Invalid direction change treated as no-op
        
        new_head = (head[0] + new_dir[0], head[1] + new_dir[1])
        in_bounds = (0 <= new_head[0] < env.GRID_WIDTH and 0 <= new_head[1] < env.GRID_HEIGHT)
        not_self = new_head not in env.snake_body
        safe = in_bounds and not_self
        dist = abs(new_head[0] - apple[0]) + abs(new_head[1] - apple[1])
        candidates.append((a0, dist, safe))
    
    # Prefer safe moves that minimize distance to apple
    safe_moves = [c for c in candidates if c[2]]
    if safe_moves:
        best_move = min(safe_moves, key=lambda x: x[1])[0]
    else:
        best_move = 0  # Default to no-op if no safe moves
    
    return [best_move, 0, 0]  # a1 and a2 unused in this environment