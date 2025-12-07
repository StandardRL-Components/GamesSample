def policy(env):
    """
    Navigate towards the exit using Manhattan distance, avoiding traps (red) and teleports (blue) unless necessary.
    Prioritize safe (green) and exit (yellow) tiles, then break ties by minimizing distance to the exit.
    Secondary actions (a1, a2) are unused in this environment and set to 0.
    """
    px, py = env.player_pos
    ex, ey = env.exit_pos
    moves = [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]  # (dx, dy, action)
    candidates = []
    
    for dx, dy, act in moves:
        nx, ny = px + dx, py + dy
        if 0 <= nx < env.GRID_SIZE and 0 <= ny < env.GRID_SIZE:
            tile_type = env.grid[ny][nx]
            dist = abs(nx - ex) + abs(ny - ey)
            # Prefer safe (green) and exit (yellow) tiles, avoid traps (red) and teleports (blue)
            if tile_type == env.TILE_YELLOW:
                priority = 0
            elif tile_type == env.TILE_GREEN:
                priority = 1
            elif tile_type == env.TILE_BLUE:
                priority = 2
            else:  # TILE_RED
                priority = 3
            candidates.append((priority, dist, act))
    
    if candidates:
        # Sort by priority (lower is better) then by distance (lower is better)
        candidates.sort(key=lambda x: (x[0], x[1]))
        best_move = candidates[0][2]
        return [best_move, 0, 0]
    return [0, 0, 0]  # No valid move, stay in place