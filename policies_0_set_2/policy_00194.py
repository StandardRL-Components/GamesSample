def policy(env):
    # Strategy: Move towards food using Manhattan distance, avoiding collisions with walls, self, and opponent.
    # This balances efficiency in reaching food with safety, maximizing reward from eating and proximity.
    current_head = env.player_body[0]
    food_pos = env.food_pos
    current_dir = env.player_direction
    
    # Define possible directions and their action codes
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
    action_map = {(-1, 0): 3, (1, 0): 4, (0, -1): 1, (0, 1): 2}  # action codes for directions
    
    # Find safe moves that don't cause immediate collisions
    safe_actions = []
    for dx, dy in dirs:
        new_pos = (current_head[0] + dx, current_head[1] + dy)
        # Check bounds and collisions
        if (0 <= new_pos[0] < env.GRID_WIDTH and 0 <= new_pos[1] < env.GRID_HEIGHT and
            new_pos not in list(env.player_body)[1:] and new_pos not in env.opponent_body):
            safe_actions.append((dx, dy))
    
    # If no safe moves, continue current direction (action 0)
    if not safe_actions:
        return [0, 0, 0]
    
    # Choose move that minimizes Manhattan distance to food
    best_move = min(safe_actions, key=lambda move: abs(current_head[0] + move[0] - food_pos[0]) + 
                                                 abs(current_head[1] + move[1] - food_pos[1]))
    
    # Convert direction to action code, default to 0 if same as current direction
    action_code = action_map.get(best_move, 0)
    return [action_code, 0, 0]