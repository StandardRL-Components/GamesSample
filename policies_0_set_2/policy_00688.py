def policy(env):
    # Strategy: Prioritize collecting pellets while avoiding non-vulnerable ghosts. 
    # When ghosts are vulnerable, chase them for high rewards. Use BFS to find nearest pellet 
    # and navigate towards it while avoiding nearby threats. Secondary actions are unused.
    from collections import deque
    
    def get_valid_moves(x, y):
        moves = []
        for dx, dy, action in [(0,-1,1), (0,1,2), (-1,0,3), (1,0,4)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.MAZE_WIDTH and 0 <= ny < env.MAZE_HEIGHT and env.maze[ny][nx] == 0:
                moves.append((nx, ny, action))
        return moves

    player_x, player_y = round(env.player.x), round(env.player.y)
    
    # Check if any ghost is vulnerable
    vulnerable_ghosts = [g for g in env.ghosts if g.state == 'vulnerable']
    if vulnerable_ghosts:
        # Chase nearest vulnerable ghost
        nearest_ghost = min(vulnerable_ghosts, key=lambda g: (g.x - player_x)**2 + (g.y - player_y)**2)
        target = (round(nearest_ghost.x), round(nearest_ghost.y))
    else:
        # Find nearest pellet using BFS
        queue = deque([(player_x, player_y, [])])
        visited = set([(player_x, player_y)])
        target = None
        
        while queue and not target:
            x, y, path = queue.popleft()
            for nx, ny, action in get_valid_moves(x, y):
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                new_path = path + [action]
                if (nx, ny) in env.pellets or (nx, ny) in env.power_pellets:
                    target = new_path[0] if new_path else 0
                    break
                queue.append((nx, ny, new_path))
        
        if target is None:
            # No pellets found, use random valid move
            valid_actions = [a for _,_,a in get_valid_moves(player_x, player_y)]
            return [valid_actions[0] if valid_actions else 0, 0, 0]
    
    # Move toward target
    best_action = 0
    min_dist = float('inf')
    for nx, ny, action in get_valid_moves(player_x, player_y):
        dist = (nx - target[0])**2 + (ny - target[1])**2
        if dist < min_dist:
            min_dist = dist
            best_action = action
    
    return [best_action, 0, 0]