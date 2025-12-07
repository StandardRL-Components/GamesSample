def policy(env):
    # Strategy: Prioritize attacking enemies in front to clear path and collect rewards, then move toward exit.
    # If health is low, avoid enemies. Collect adjacent gold when safe. Break ties by moving horizontally first.
    px, py = env.player_pos
    ex, ey = env.exit_pos
    move_dirs = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
    
    # Check for attack opportunity in last moved direction
    attack_x = px + env.player_last_move_dir[0]
    attack_y = py + env.player_last_move_dir[1]
    for enemy in env.enemies:
        if (attack_x, attack_y) == enemy['pos'] and env.player_health > enemy['attack']:
            return [0, 1, 0]
    
    # Evaluate safe moves (non-wall, no enemy)
    safe_moves = []
    for move, (dx, dy) in move_dirs.items():
        nx, ny = px + dx, py + dy
        if env.grid[nx][ny] != env.TILE_WALL:
            safe = True
            for enemy in env.enemies:
                if (nx, ny) == enemy['pos']:
                    safe = False
                    break
            if safe:
                safe_moves.append((move, nx, ny))
    
    # If low health, avoid enemies even if not adjacent
    if env.player_health <= 2:
        best_move = None
        max_dist = -1
        for move, nx, ny in safe_moves:
            min_enemy_dist = min(abs(nx - e['pos'][0]) + abs(ny - e['pos'][1]) for e in env.enemies)
            if min_enemy_dist > max_dist:
                max_dist = min_enemy_dist
                best_move = move
        if best_move is not None:
            return [best_move, 0, 0]
    
    # Collect adjacent gold
    for move, nx, ny in safe_moves:
        if (nx, ny) in env.gold_pieces:
            return [move, 0, 0]
    
    # Move toward exit
    if safe_moves:
        best_move = None
        min_dist = float('inf')
        for move, nx, ny in safe_moves:
            dist = abs(nx - ex) + abs(ny - ey)
            if dist < min_dist:
                min_dist = dist
                best_move = move
        return [best_move, 0, 0]
    
    return [0, 0, 0]