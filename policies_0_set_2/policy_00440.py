def policy(env):
    # This policy maximizes score by always targeting the largest available fruit cluster to maximize cascade potential and points per move.
    # If no valid cluster exists, it triggers a reshuffle. Movement is optimized to reach the target cluster efficiently.
    visited = set()
    clusters = []
    for y in range(env.GRID_HEIGHT):
        for x in range(env.GRID_WIDTH):
            if (x, y) not in visited and env.grid[y][x] != 0:
                cluster = env._find_cluster_at(x, y)
                if len(cluster) >= env.MIN_CLUSTER_SIZE:
                    clusters.append(cluster)
                visited.update(cluster)
    
    if not clusters:
        return [0, 1, 0]
    
    best_cluster = max(clusters, key=len)
    cursor_pos = (env.cursor_pos[0], env.cursor_pos[1])
    if cursor_pos in best_cluster:
        return [0, 1, 0]
    
    min_dist = float('inf')
    target = None
    for cell in best_cluster:
        dist = abs(cell[0] - env.cursor_pos[0]) + abs(cell[1] - env.cursor_pos[1])
        if dist < min_dist:
            min_dist = dist
            target = cell
    
    dx = target[0] - env.cursor_pos[0]
    dy = target[1] - env.cursor_pos[1]
    action0 = 0
    if abs(dx) > abs(dy):
        action0 = 4 if dx > 0 else 3
    else:
        action0 = 2 if dy > 0 else 1
    return [action0, 0, 0]