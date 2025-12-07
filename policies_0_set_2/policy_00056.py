def policy(env):
    """
    Strategy: Navigate towards exit while avoiding enemies. Attack only when adjacent to enemies to eliminate threats.
    Prioritize moves that reduce Manhattan distance to exit while avoiding positions adjacent to enemies to prevent damage.
    This balances progress with safety, maximizing reward by completing levels efficiently while minimizing health loss.
    """
    player_pos = env.player_pos
    exit_pos = env.exit_pos
    enemies = env.enemies
    
    # Calculate movement directions towards exit
    dx = exit_pos[0] - player_pos[0]
    dy = exit_pos[1] - player_pos[1]
    
    # Check for adjacent enemies
    adjacent_enemies = []
    for enemy in enemies:
        if abs(enemy[0] - player_pos[0]) + abs(enemy[1] - player_pos[1]) == 1:
            adjacent_enemies.append(enemy)
    
    # Attack if adjacent to enemies
    if adjacent_enemies:
        return [0, 1, 0]
    
    # Move towards exit while avoiding positions adjacent to enemies
    possible_moves = []
    for move, (dx, dy) in enumerate([(0,0), (0,-1), (0,1), (-1,0), (1,0)], 0):
        new_x, new_y = player_pos[0] + dx, player_pos[1] + dy
        if not (0 <= new_x < env.GRID_COLS and 0 <= new_y < env.GRID_ROWS):
            continue
            
        # Check if new position is safe (not adjacent to enemies)
        safe = True
        for enemy in enemies:
            if abs(enemy[0] - new_x) + abs(enemy[1] - new_y) == 1:
                safe = False
                break
                
        if safe:
            dist_to_exit = abs(exit_pos[0] - new_x) + abs(exit_pos[1] - new_y)
            possible_moves.append((move, dist_to_exit))
    
    # Choose best safe move towards exit
    if possible_moves:
        best_move = min(possible_moves, key=lambda x: x[1])[0]
        return [best_move, 0, 0]
    
    # If no safe moves, move towards exit anyway
    if abs(dx) > abs(dy):
        move_dir = 4 if dx > 0 else 3
    else:
        move_dir = 2 if dy > 0 else 1 if dy < 0 else 0
    return [move_dir, 0, 0]