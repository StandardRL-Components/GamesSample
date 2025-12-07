def policy(env):
    # This policy uses a greedy approach to navigate the snake towards food while avoiding collisions.
    # It prioritizes moves that reduce Manhattan distance to food, checks for safety (walls/body),
    # and avoids reversals. Secondary actions are unused in Snake, so a1/a2 are set to 0.
    head = env.snake_body[0]
    food = env.food_pos
    current_dir = env.direction
    moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    move_actions = [1, 2, 3, 4]
    opposite = (-current_dir[0], -current_dir[1])
    
    safe_moves = []
    for idx, move in enumerate(moves):
        if move == opposite:
            continue
        new_head = (head[0] + move[0], head[1] + move[1])
        if not (0 <= new_head[0] < env.GRID_WIDTH and 0 <= new_head[1] < env.GRID_HEIGHT):
            continue
        obstacles = set(env.snake_body)
        if new_head != food:
            obstacles.discard(env.snake_body[-1])
        if new_head in obstacles:
            continue
        safe_moves.append((move_actions[idx], abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])))
    
    if safe_moves:
        safe_moves.sort(key=lambda x: x[1])
        return [safe_moves[0][0], 0, 0]
    
    for idx, move in enumerate(moves):
        if move != opposite:
            return [move_actions[idx], 0, 0]
    return [0, 0, 0]