def policy(env):
    """
    Uses BFS to find the shortest path to the target gem, avoiding obstacles.
    Prioritizes reaching the target efficiently to maximize rewards from gem collection
    while minimizing move penalties. Secondary actions are unused in this environment.
    """
    if env.game_over:
        return [0, 0, 0]
    
    start = env.gem_pos
    goal = env.target_pos
    obstacles = set(env.obstacles)
    grid_size = env.GRID_SIZE
    
    # BFS to find shortest path
    queue = [start]
    visited = {start}
    parent = {}
    found = False
    
    while queue:
        current = queue.pop(0)
        if current == goal:
            found = True
            break
        x, y = current
        for dx, dy in [(0,-1), (0,1), (-1,0), (1,0)]:
            neighbor = (x+dx, y+dy)
            if (0 <= neighbor[0] < grid_size[0] and 
                0 <= neighbor[1] < grid_size[1] and 
                neighbor not in obstacles and 
                neighbor not in visited):
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    if not found:
        return [0, 0, 0]
    
    # Reconstruct path to find first move
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent[current]
    next_pos = path[-1]
    
    # Determine movement direction
    dx = next_pos[0] - start[0]
    dy = next_pos[1] - start[1]
    if dx == 1:
        move = 4  # right
    elif dx == -1:
        move = 3  # left
    elif dy == 1:
        move = 2  # down
    elif dy == -1:
        move = 1  # up
    else:
        move = 0
    
    return [move, 0, 0]