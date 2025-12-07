def policy(env):
    # Strategy: Navigate directly to the goal using Manhattan distance, prioritizing moves that minimize distance.
    # Avoid jumping when mid-air and only move to tiles with positive timers (existing tiles).
    if env.is_jumping:
        return [0, 0, 0]
    px, py = env.player_pos[0], env.player_pos[1]
    gx, gy = env.goal_pos[0], env.goal_pos[1]
    best_move = 0
    best_dist = abs(px - gx) + abs(py - gy)
    moves = [(1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0)]
    for move, dx, dy in moves:
        nx, ny = px + dx, py + dy
        if 0 <= nx < env.GRID_SIZE and 0 <= ny < env.GRID_SIZE and env.tile_timers[nx, ny] > 0:
            dist = abs(nx - gx) + abs(ny - gy)
            if dist < best_dist:
                best_dist = dist
                best_move = move
    return [best_move, 0, 0]