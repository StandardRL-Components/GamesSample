def policy(env):
    """
    Uses BFS to find the shortest path to the nearest crystal, avoiding walls and boundaries.
    Prioritizes movement towards closest crystal to maximize collection rate under time constraints.
    Falls back to valid random movement if no crystal found within search depth.
    """
    player_pos = (int(env.player_pos.x), int(env.player_pos.y))
    crystal_positions = [(int(c.x), int(c.y)) for c in env.crystals]
    if not crystal_positions:
        return [0, 0, 0]
    
    visited = set()
    queue = [(player_pos[0], player_pos[1], 0, None)]
    visited.add(player_pos)
    
    while queue:
        x, y, steps, first_move = queue.pop(0)
        
        if steps > 100:
            continue
            
        if (x, y) in crystal_positions:
            if first_move is not None:
                return [first_move, 0, 0]
            return [0, 0, 0]
        
        for dx, dy, move in [(0, -1, 1), (0, 1, 2), (-1, 0, 3), (1, 0, 4)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT):
                continue
            wall = tuple(sorted([(x, y), (nx, ny)]))
            if wall in env.walls:
                continue
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                next_first = first_move if first_move is not None else move
                queue.append((nx, ny, steps + 1, next_first))
    
    for move in [1, 2, 3, 4]:
        dx, dy = (0, -1) if move == 1 else (0, 1) if move == 2 else (-1, 0) if move == 3 else (1, 0)
        nx, ny = player_pos[0] + dx, player_pos[1] + dy
        if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT):
            continue
        wall = tuple(sorted([player_pos, (nx, ny)]))
        if wall not in env.walls:
            return [move, 0, 0]
            
    return [0, 0, 0]