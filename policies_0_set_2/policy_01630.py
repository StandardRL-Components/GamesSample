def policy(env):
    # Uses BFS to find the shortest path to the nearest gem, then returns the first move direction.
    # This minimizes moves spent per gem, crucial due to limited moves. Immediate collection is prioritized.
    if not env.gem_positions:
        return [0, 0, 0]
    if env.robot_pos in env.gem_positions:
        return [0, 0, 0]
    
    queue = [env.robot_pos]
    visited = {env.robot_pos}
    parent = {env.robot_pos: None}
    index = 0
    found_gem = None
    
    while index < len(queue):
        current = queue[index]
        index += 1
        if current in env.gem_positions:
            found_gem = current
            break
        x, y = current
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            new_pos = (nx, ny)
            if (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT and
                    new_pos not in env.obstacle_positions and new_pos not in visited):
                visited.add(new_pos)
                queue.append(new_pos)
                parent[new_pos] = current
                
    if found_gem is None:
        return [0, 0, 0]
        
    path = []
    node = found_gem
    while node != env.robot_pos:
        path.append(node)
        node = parent[node]
    first_step = path[-1]
    
    rx, ry = env.robot_pos
    fx, fy = first_step
    if fx == rx and fy == ry - 1:
        move = 1
    elif fx == rx and fy == ry + 1:
        move = 2
    elif fx == rx - 1 and fy == ry:
        move = 3
    elif fx == rx + 1 and fy == ry:
        move = 4
    else:
        move = 0
        
    return [move, 0, 0]