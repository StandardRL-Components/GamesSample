def policy(env):
    """
    Strategy: Use BFS to find the shortest path to the target number (highest value).
    Prioritize reaching the target before running out of moves, as the win reward (100)
    dominates step rewards (±0.1). Movement actions are derived from the BFS path.
    """
    if env.game_over:
        return [0, 0, 0]
    
    player_pos = env.player_pos
    target_pos = env.target_pos
    
    if player_pos == target_pos:
        return [0, 0, 0]
    
    queue = [(player_pos, [player_pos])]
    visited = {player_pos}
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    while queue:
        (x, y), path = queue.pop(0)
        if (x, y) == target_pos:
            next_cell = path[1]
            dx = next_cell[0] - player_pos[0]
            dy = next_cell[1] - player_pos[1]
            if dx == 0 and dy == -1:
                return [1, 0, 0]
            elif dx == 0 and dy == 1:
                return [2, 0, 0]
            elif dx == -1 and dy == 0:
                return [3, 0, 0]
            elif dx == 1 and dy == 0:
                return [4, 0, 0]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [(nx, ny)]))
    
    return [0, 0, 0]