def policy(env):
    """
    Greedy gem collection by moving towards the nearest gem using Manhattan distance.
    Prioritizes immediate collection when adjacent, otherwise moves to minimize distance.
    Secondary actions (a1, a2) are unused in this environment and set to 0.
    """
    px, py = env.player_pos
    gems = list(env.gems)
    if not gems:
        return [0, 0, 0]
    
    best_move = 0
    best_dist = float('inf')
    moves = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
    
    for move, (dx, dy) in enumerate(moves):
        nx, ny = px + dx, py + dy
        if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT):
            continue
            
        min_gem_dist = min(abs(nx - gx) + abs(ny - gy) for gx, gy in gems)
        if min_gem_dist < best_dist:
            best_dist = min_gem_dist
            best_move = move
            
    return [best_move, 0, 0]