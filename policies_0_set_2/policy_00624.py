def policy(env):
    """
    Uses BFS to find the shortest path to the exit, avoiding obstacles. 
    Prioritizes minimizing moves to maximize reward (reaching exit gives +10, 
    each move costs -0.1). Avoids restart action (a1=1) to prevent penalty.
    """
    if env.game_state != "playing":
        return [0, 0, 0]
    
    grid_size = env.grid_size
    obstacles_set = set(tuple(obs) for obs in env.obstacles)
    start = tuple(env.robot_pos)
    goal = tuple(env.exit_pos)
    
    if start == goal:
        return [0, 0, 0]
    
    queue = [start]
    visited = {start}
    parent = {}
    directions = [(0,-1), (0,1), (-1,0), (1,0)]
    
    while queue:
        current = queue.pop(0)
        if current == goal:
            break
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                neighbor not in obstacles_set and neighbor not in visited):
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    if goal not in parent:
        return [0, 0, 0]
    
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent[current]
    path.reverse()
    
    next_cell = path[0]
    dx = next_cell[0] - start[0]
    dy = next_cell[1] - start[1]
    
    if dx == 0 and dy == -1:
        movement = 1
    elif dx == 0 and dy == 1:
        movement = 2
    elif dx == -1 and dy == 0:
        movement = 3
    elif dx == 1 and dy == 0:
        movement = 4
    else:
        movement = 0
        
    return [movement, 0, 0]