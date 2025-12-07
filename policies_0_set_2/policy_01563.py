def policy(env):
    """
    Strategy: Navigate the snake towards food using Manhattan distance, avoiding collisions.
    Prioritize moves that are safe (not leading to wall or body collision) and minimize distance to food.
    If no safe move exists, continue current direction (a0=0) to delay collision.
    Secondary actions (a1, a2) are unused in Snake and set to 0.
    """
    current_direction = env.direction
    reverse_map = {1: 2, 2: 1, 3: 4, 4: 3}
    reverse_dir = reverse_map.get(current_direction, current_direction)
    head = env.snake_pos[0]
    food = env.food_pos

    def is_safe(move):
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[move]
        new_head = (head[0] + dx, head[1] + dy)
        if not (0 <= new_head[0] < env.GRID_WIDTH and 0 <= new_head[1] < env.GRID_HEIGHT):
            return False
        if new_head == food:
            return True
        return new_head not in list(env.snake_pos)[:-1]

    allowed_moves = [d for d in [1, 2, 3, 4] if d != reverse_dir]
    safe_moves = [move for move in allowed_moves if is_safe(move)]

    if safe_moves:
        best_move = min(safe_moves, key=lambda move: abs(head[0] + {1:0,2:0,3:-1,4:1}[move] - food[0]) + abs(head[1] + {1:-1,2:1,3:0,4:0}[move] - food[1]))
        return [best_move, 0, 0]
    else:
        return [0, 0, 0]