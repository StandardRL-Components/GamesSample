def policy(env):
    """
    Strategy: Prioritize attacking adjacent enemies to gain XP and rewards, then move towards stairs or nearest enemy.
    Avoid unnecessary movement to minimize damage taken. Defend only when low health and surrounded by threats.
    """
    player_x, player_y = env.player['x'], env.player['y']
    
    # Check for adjacent enemies to attack
    for enemy in env.enemies:
        if abs(player_x - enemy['x']) <= 1 and abs(player_y - enemy['y']) <= 1:
            return [0, 1, 0]  # Attack adjacent enemy
    
    # If no adjacent enemies, move towards target
    target = None
    if env.enemies:
        # Find nearest enemy
        min_dist = float('inf')
        for enemy in env.enemies:
            dist = abs(player_x - enemy['x']) + abs(player_y - enemy['y'])
            if dist < min_dist:
                min_dist = dist
                target = (enemy['x'], enemy['y'])
    elif env.stairs_pos:
        target = env.stairs_pos  # Move to stairs if no enemies
    
    if target:
        tx, ty = target
        dx = tx - player_x
        dy = ty - player_y
        
        # Prefer movement direction that reduces largest distance component
        if abs(dx) > abs(dy):
            move_dir = 4 if dx > 0 else 3  # Right or left
        else:
            move_dir = 2 if dy > 0 else 1  # Down or up
        
        # Check if move is valid (not into wall or enemy)
        new_x, new_y = player_x, player_y
        if move_dir == 1: new_y -= 1
        elif move_dir == 2: new_y += 1
        elif move_dir == 3: new_x -= 1
        elif move_dir == 4: new_x += 1
        
        if (0 <= new_x < env.DUNGEON_WIDTH and 
            0 <= new_y < env.DUNGEON_HEIGHT and 
            env.dungeon_map[new_y][new_x] == 0 and 
            not any(e['x'] == new_x and e['y'] == new_y for e in env.enemies)):
            return [move_dir, 0, 0]
    
    # Default: defend if no better action
    return [0, 0, 0]