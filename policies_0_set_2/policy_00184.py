def policy(env):
    # Strategy: Analyze obstacles in the row above the player to find the safe gap.
    # Move horizontally if needed to align with the gap, otherwise jump straight up to avoid penalty.
    # This maximizes upward progress while minimizing collisions and unnecessary moves.
    player_x, player_y = env.player_pos
    next_row = player_y + 1
    
    # Find obstacles in the next row
    obstacles_in_row = []
    for obs in env.obstacles:
        if obs['y'] == next_row:
            obstacles_in_row.append(obs)
    
    # If no obstacles, jump straight up
    if not obstacles_in_row:
        return [1, 0, 0]
    
    # Calculate blocked columns
    blocked = set()
    for obs in obstacles_in_row:
        for col in range(obs['x'], obs['x'] + obs['w']):
            blocked.add(col)
    
    # Find safe gap (free columns)
    free_cols = [col for col in range(env.GRID_COLS) if col not in blocked]
    if not free_cols:
        return [1, 0, 0]
    
    gap_min, gap_max = min(free_cols), max(free_cols)
    
    # Move to align with gap
    if player_x < gap_min:
        return [4, 0, 0]  # Move right
    elif player_x > gap_max:
        return [3, 0, 0]  # Move left
    else:
        return [1, 0, 0]  # Jump straight up