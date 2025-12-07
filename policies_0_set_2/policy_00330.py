def policy(env):
    # Strategy: Prioritize building turrets in center rows (y=4-6) for maximum coverage, then walls for protection.
    # Move cursor to nearest valid empty cell and place appropriate block type (turrets on even rows, walls on odd).
    # If enemy is very close (y>=3), build wall in its path immediately to block advance.
    cursor_pos = env.cursor_pos
    blocks = env.blocks
    enemies = env.enemies
    
    # Check for immediate threats (enemies near fortress)
    for enemy in enemies:
        if enemy.pos[1] >= 3:  # Enemy close to build area
            target_x = int(round(enemy.pos[0]))
            target_y = 4  # Build at fortress line
            if (target_x, target_y) not in blocks:
                dx = target_x - cursor_pos[0]
                dy = target_y - cursor_pos[1]
                if dx == 0 and dy == 0:
                    current_type = env.available_block_types[env.selected_block_type_idx]
                    if current_type != "WALL":
                        return [0, 0, 1]  # Cycle to wall
                    return [0, 1, 0]  # Place wall
                if abs(dx) > abs(dy):
                    return [4 if dx > 0 else 3, 0, 0]
                return [2 if dy > 0 else 1, 0, 0]
    
    # Build pattern: turrets on even rows (y=4,6,8,...), walls on odd rows (y=5,7,9,...)
    for y in range(4, 16):
        for x in range(0, 16):
            if (x, y) not in blocks:
                desired_type = "TURRET" if y % 2 == 0 else "WALL"
                dx = x - cursor_pos[0]
                dy = y - cursor_pos[1]
                if dx == 0 and dy == 0:
                    current_type = env.available_block_types[env.selected_block_type_idx]
                    if current_type != desired_type:
                        return [0, 0, 1]  # Cycle block type
                    return [0, 1, 0]  # Place block
                if abs(dx) > abs(dy):
                    return [4 if dx > 0 else 3, 0, 0]
                return [2 if dy > 0 else 1, 0, 0]
    
    return [0, 0, 0]  # Default no-op if grid full