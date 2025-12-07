def policy(env):
    # Use BFS to find the shortest path to the exit, returning the first move direction.
    # This maximizes reward by minimizing steps to reach the exit, avoiding walls and dead ends.
    if env.player_pos == env.exit_pos:
        return [0, 0, 0]
    
    moves = [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]
    start = tuple(env.player_pos)
    goal = tuple(env.exit_pos)
    visited = set([start])
    queue = [start]
    index = 0
    first_move = {}
    
    while index < len(queue):
        current = queue[index]
        index += 1
        if current == goal:
            break
            
        for dr, dc, move_dir in moves:
            r, c = current[0] + dr, current[1] + dc
            neighbor = (r, c)
            if (r < 0 or r >= env.maze.shape[0] or c < 0 or c >= env.maze.shape[1] or
                env.maze[r, c] != 0 or neighbor in visited):
                continue
            visited.add(neighbor)
            queue.append(neighbor)
            if current == start:
                first_move[neighbor] = move_dir
            else:
                first_move[neighbor] = first_move[current]
                
    if goal in first_move:
        return [first_move[goal], 0, 0]
    else:
        best_move = 0
        min_dist = float('inf')
        for dr, dc, move_dir in moves:
            r, c = env.player_pos[0] + dr, env.player_pos[1] + dc
            if (0 <= r < env.maze.shape[0] and 0 <= c < env.maze.shape[1] and 
                env.maze[r, c] == 0):
                dist = abs(r - env.exit_pos[0]) + abs(c - env.exit_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    best_move = move_dir
        return [best_move, 0, 0]