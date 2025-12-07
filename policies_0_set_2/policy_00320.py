def policy(env):
    # Greedy policy that moves towards the nearest fruit using Manhattan distance to maximize collection rate and minimize time.
    # This minimizes time spent moving and maximizes the shaping reward for moving closer to fruit.
    if not env.fruits:
        return [0, 0, 0]
    
    player_pos = env.player_grid_pos
    min_dist = float('inf')
    nearest_fruit = None
    for fruit in env.fruits:
        dist = abs(fruit['pos'][0] - player_pos[0]) + abs(fruit['pos'][1] - player_pos[1])
        if dist < min_dist:
            min_dist = dist
            nearest_fruit = fruit
    
    if min_dist == 0:
        return [0, 0, 0]
    
    best_action = 0
    best_improvement = 0
    moves = [1, 2, 3, 4]
    for move in moves:
        new_pos = list(player_pos)
        if move == 1:
            new_pos[1] -= 1
        elif move == 2:
            new_pos[1] += 1
        elif move == 3:
            new_pos[0] -= 1
        elif move == 4:
            new_pos[0] += 1
        
        if new_pos[0] < 0 or new_pos[0] >= env.GRID_COLS or new_pos[1] < 0 or new_pos[1] >= env.GRID_ROWS:
            continue
            
        new_dist = abs(nearest_fruit['pos'][0] - new_pos[0]) + abs(nearest_fruit['pos'][1] - new_pos[1])
        improvement = min_dist - new_dist
        if improvement > best_improvement:
            best_improvement = improvement
            best_action = move
    
    return [best_action, 0, 0]