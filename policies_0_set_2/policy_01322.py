def policy(env):
    # Strategy: Greedy Manhattan distance to food, avoiding walls and body. 
    # Prioritize moves that reduce distance to food, breaking ties by move order.
    # Use current direction as fallback when no safe move exists.
    head = env.snake_body[0]
    food = env.food_pos
    moves = {0: env.direction, 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    opposite = (-env.direction[0], -env.direction[1])
    allowed = [0] + [i for i in range(1, 5) if moves[i] != opposite]
    safe_moves = []
    for move_id in allowed:
        dx, dy = moves[move_id]
        new_pos = (head[0] + dx, head[1] + dy)
        if (0 <= new_pos[0] < env.GRID_WIDTH and 
            0 <= new_pos[1] < env.GRID_HEIGHT and 
            new_pos not in env.snake_body):
            safe_moves.append(move_id)
    if safe_moves:
        best_move = min(safe_moves, key=lambda m: abs(head[0] + moves[m][0] - food[0]) + abs(head[1] + moves[m][1] - food[1]))
        return [best_move, 0, 0]
    return [0, 0, 0]