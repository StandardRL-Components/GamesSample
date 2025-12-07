def policy(env):
    # Strategy: Move towards the nearest crystal to collect it as quickly as possible. Ignore swap and shift actions since moving directly onto crystals is more efficient. Use squared distance to compare moves, breaking ties by move index order.
    if env.game_over:
        return [0, 0, 0]
    px, py = env.player_pos.x, env.player_pos.y
    if not env.crystals:
        return [0, 0, 0]
    moves = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
    best_action = 0
    best_sq_dist = float('inf')
    for action_index in [1, 2, 3, 4]:
        dx, dy = moves[action_index]
        nx, ny = px + dx, py + dy
        if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT):
            continue
        min_sq_dist = float('inf')
        for crystal in env.crystals:
            cdx, cdy = crystal.x - nx, crystal.y - ny
            sq_dist = cdx * cdx + cdy * cdy
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
        if min_sq_dist < best_sq_dist:
            best_sq_dist = min_sq_dist
            best_action = action_index
    return [best_action, 0, 0]