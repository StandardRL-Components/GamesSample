def policy(env):
    # Strategy: Greedy pathfinding with collision avoidance. Prioritize moving toward food using wrapped Manhattan distance while avoiding self-collision and invalid 180° turns. Break ties by action priority (right, down, left, up, none).
    if env.game_over:
        return [0, 0, 0]
    
    head = env.snake_body[-1]
    food = env.food_pos
    current_dir = env.snake_direction
    w, h = env.GRID_WIDTH, env.GRID_HEIGHT
    body_without_head = list(env.snake_body)[:-1]
    
    best_action = 0
    best_dist = float('inf')
    
    for move, dir_vec in enumerate([(0,0), (0,-1), (0,1), (-1,0), (1,0)], 0):
        if move == 0:
            new_dir = current_dir
        else:
            new_dir = dir_vec
            if new_dir[0] == -current_dir[0] and new_dir[1] == -current_dir[1]:
                continue
        
        new_x = (head[0] + new_dir[0]) % w
        new_y = (head[1] + new_dir[1]) % h
        new_head = (new_x, new_y)
        
        if new_head in body_without_head:
            continue
            
        dx = min(abs(new_x - food[0]), w - abs(new_x - food[0]))
        dy = min(abs(new_y - food[1]), h - abs(new_y - food[1]))
        dist = dx + dy
        
        if dist < best_dist or (dist == best_dist and move < best_action):
            best_dist = dist
            best_action = move
    
    return [best_action, 0, 0]