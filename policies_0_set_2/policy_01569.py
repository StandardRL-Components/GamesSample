def policy(env):
    # Strategy: Choose the safest move that minimizes Manhattan distance to food, avoiding walls and self-collision.
    # If multiple moves are equally good, break ties by preferring current direction, then up, down, left, right.
    # Always use a1=0 and a2=0 as they are unused in this environment.
    head = env.snake_pos[0]
    food = env.food_pos
    current_dir = env.snake_direction
    opposite_dir = (-current_dir[0], -current_dir[1])
    moves = [1, 2, 3, 4]  # up, down, left, right
    vectors = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    snake_body = set(env.snake_pos)
    
    candidate_actions = [0]  # no-op (continue current direction)
    for move in moves:
        if vectors[move] != opposite_dir:
            candidate_actions.append(move)
    
    safe_actions = []
    for a in candidate_actions:
        if a == 0:
            new_dir = current_dir
        else:
            new_dir = vectors[a]
        new_head = (head[0] + new_dir[0], head[1] + new_dir[1])
        if (0 <= new_head[0] < env.GRID_WIDTH and 
            0 <= new_head[1] < env.GRID_HEIGHT and 
            new_head not in snake_body):
            safe_actions.append(a)
    
    if not safe_actions:
        return [candidate_actions[0], 0, 0]
    
    best_action = None
    min_dist = float('inf')
    for a in safe_actions:
        if a == 0:
            new_dir = current_dir
        else:
            new_dir = vectors[a]
        new_head = (head[0] + new_dir[0], head[1] + new_dir[1])
        dist = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
        if dist < min_dist:
            min_dist = dist
            best_action = a
        elif dist == min_dist:
            if best_action is None or a < best_action:
                best_action = a
    
    return [best_action, 0, 0]