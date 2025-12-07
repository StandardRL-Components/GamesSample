def policy(env):
    # Strategy: Use BFS to compute optimal path to exit, then choose move that minimizes
    # distance while avoiding ghosts. Prioritize exit proximity with ghost safety margins.
    if env.game_over:
        return [0, 0, 0]
    
    px, py = env.player_pos
    ex, ey = env.exit_pos
    
    # BFS to compute distances from exit
    dist = {}
    queue = [(ex, ey)]
    dist[(ex, ey)] = 0
    while queue:
        x, y = queue.pop(0)
        for dx, dy, dir in [(0,-1,'N'), (0,1,'S'), (-1,0,'W'), (1,0,'E')]:
            nx, ny = x+dx, y+dy
            if (0<=nx<env.MAZE_W and 0<=ny<env.MAZE_H and (nx,ny) not in dist and 
                dir in env.maze.get((x,y), set())):
                dist[(nx,ny)] = dist[(x,y)] + 1
                queue.append((nx,ny))
    
    best_score = -10**9
    best_action = [0, 0, 0]
    moves = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]
    
    for a0 in range(5):
        dx, dy = moves[a0]
        nx, ny = px+dx, py+dy
        if not (0<=nx<env.MAZE_W and 0<=ny<env.MAZE_H):
            continue
        if a0 != 0 and moves[a0][1] != 0 and ('N' if dy<0 else 'S') not in env.maze.get((px,py), set()):
            continue
        if a0 != 0 and moves[a0][0] != 0 and ('W' if dx<0 else 'E') not in env.maze.get((px,py), set()):
            continue
        
        # Calculate safety score
        ghost_penalty = 0
        for ghost in env.ghosts:
            gx, gy = ghost["pos"]
            d = abs(nx-gx) + abs(ny-gy)
            if d == 0:
                ghost_penalty -= 1000
            elif d == 1:
                ghost_penalty -= 100
            elif d == 2:
                ghost_penalty -= 10
        
        # Prefer moves that reduce distance to exit
        dist_score = -dist.get((nx,ny), 1000)
        
        total_score = dist_score + ghost_penalty
        if total_score > best_score:
            best_score = total_score
            best_action = [a0, 0, 0]
            
    return best_action