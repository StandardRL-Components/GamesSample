def policy(env):
    # Strategy: Prioritize attacking adjacent enemies when facing them, otherwise turn to face enemies.
    # If no enemies are adjacent, move towards the exit while avoiding walls and enemies.
    # This maximizes reward by eliminating threats (+1 per kill) and reaching the exit (+100).
    player_pos = env.player_pos
    exit_pos = env.exit_pos
    enemies = env.enemies
    dungeon = env.dungeon
    last_dir = env.player_last_move_dir
    
    # Check adjacent cells for enemies
    adjacent_positions = [
        (player_pos[0], player_pos[1] - 1),  # Up
        (player_pos[0], player_pos[1] + 1),  # Down
        (player_pos[0] - 1, player_pos[1]),  # Left
        (player_pos[0] + 1, player_pos[1])   # Right
    ]
    
    # Check if currently facing an enemy
    attack_pos = (player_pos[0] + last_dir[0], player_pos[1] + last_dir[1])
    for enemy in enemies:
        if enemy['pos'] == attack_pos:
            return [0, 1, 0]  # Attack without moving
    
    # Check for any adjacent enemy and turn towards it
    for i, pos in enumerate(adjacent_positions, 1):
        for enemy in enemies:
            if enemy['pos'] == pos:
                return [i, 0, 0]  # Move direction i (1-4) to face enemy
    
    # Move towards exit using Manhattan distance
    dx = exit_pos[0] - player_pos[0]
    dy = exit_pos[1] - player_pos[1]
    
    # Prefer horizontal movement if farther away horizontally
    if abs(dx) > abs(dy):
        move_dir = 4 if dx > 0 else 3  # Right or left
    else:
        move_dir = 2 if dy > 0 else 1  # Down or up
    
    # Check if preferred move is valid (not wall or enemy)
    new_pos = adjacent_positions[move_dir - 1]
    if (0 <= new_pos[0] < env.GRID_WIDTH and 
        0 <= new_pos[1] < env.GRID_HEIGHT and 
        dungeon[new_pos[0], new_pos[1]] == 0 and
        not any(enemy['pos'] == new_pos for enemy in enemies)):
        return [move_dir, 0, 0]
    
    # Fallback: try any valid move
    for i in range(1, 5):
        new_pos = adjacent_positions[i - 1]
        if (0 <= new_pos[0] < env.GRID_WIDTH and 
            0 <= new_pos[1] < env.GRID_HEIGHT and 
            dungeon[new_pos[0], new_pos[1]] == 0 and
            not any(enemy['pos'] == new_pos for enemy in enemies)):
            return [i, 0, 0]
    
    return [0, 0, 0]  # No valid move