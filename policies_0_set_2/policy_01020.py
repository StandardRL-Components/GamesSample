def policy(env):
    # Strategy: Use BFS to find shortest path to exit while avoiding monsters. If next step is unsafe (monster present or moving there), wait or choose safest alternative move that minimizes distance to exit.
    from collections import deque
    
    def bfs(start, goal):
        queue = deque([(start, [])])
        visited = set([start])
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal:
                return path + [(x, y)]
            for dx, dy, move in [(0,-1,1), (0,1,2), (-1,0,3), (1,0,4)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < env.maze_dim and 0 <= ny < env.maze_dim:
                    if move == 1 and not env.maze[y][x]['top'] and (nx, ny) not in visited:
                        queue.append(((nx, ny), path + [(x, y)]))
                        visited.add((nx, ny))
                    elif move == 2 and not env.maze[y][x]['bottom'] and (nx, ny) not in visited:
                        queue.append(((nx, ny), path + [(x, y)]))
                        visited.add((nx, ny))
                    elif move == 3 and not env.maze[y][x]['left'] and (nx, ny) not in visited:
                        queue.append(((nx, ny), path + [(x, y)]))
                        visited.add((nx, ny))
                    elif move == 4 and not env.maze[y][x]['right'] and (nx, ny) not in visited:
                        queue.append(((nx, ny), path + [(x, y)]))
                        visited.add((nx, ny))
        return []
    
    player_pos = env.player_pos
    exit_pos = env.exit_pos
    
    if player_pos == exit_pos:
        return [0, 0, 0]
    
    path = bfs(player_pos, exit_pos)
    if not path:
        return [0, 0, 0]
    
    next_step = path[1] if len(path) > 1 else path[0]
    
    # Check if next step is safe from monsters
    safe = True
    for monster in env.monsters:
        if monster['pos'] == next_step:
            safe = False
            break
        if monster['move_counter'] + monster['speed'] >= 1:
            next_idx = monster['path_idx'] + monster['path_dir']
            if next_idx < 0 or next_idx >= len(monster['path']):
                next_idx = monster['path_idx'] - monster['path_dir']
            if monster['path'][next_idx] == next_step:
                safe = False
                break
    
    if safe:
        dx = next_step[0] - player_pos[0]
        dy = next_step[1] - player_pos[1]
        if dx == 1:
            return [4, 0, 0]
        elif dx == -1:
            return [3, 0, 0]
        elif dy == 1:
            return [2, 0, 0]
        elif dy == -1:
            return [1, 0, 0]
    
    # If unsafe, wait or choose alternative move
    moves = []
    x, y = player_pos
    if y > 0 and not env.maze[y][x]['top']:
        moves.append((1, (x, y-1)))
    if y < env.maze_dim-1 and not env.maze[y][x]['bottom']:
        moves.append((2, (x, y+1)))
    if x > 0 and not env.maze[y][x]['left']:
        moves.append((3, (x-1, y)))
    if x < env.maze_dim-1 and not env.maze[y][x]['right']:
        moves.append((4, (x+1, y)))
    
    # Find safest move (furthest from monsters)
    best_move = 0
    best_score = -float('inf')
    for move, pos in moves:
        min_monster_dist = min(
            abs(pos[0]-m['pos'][0]) + abs(pos[1]-m['pos'][1])
            for m in env.monsters
        )
        exit_dist = abs(pos[0]-exit_pos[0]) + abs(pos[1]-exit_pos[1])
        score = min_monster_dist - exit_dist
        if score > best_score:
            best_score = score
            best_move = move
    
    return [best_move, 0, 0]