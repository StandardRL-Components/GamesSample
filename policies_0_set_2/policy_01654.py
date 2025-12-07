def policy(env):
    # Strategy: Use BFS to find the shortest path to the nearest fruit, then move toward it.
    # This maximizes reward by prioritizing immediate fruit collection (+10) while minimizing steps due to the time limit.
    if env.game_over:
        return [0, 0, 0]
    
    player_x, player_y = env.player_pos
    if [player_x, player_y] in env.fruits:
        return [0, 0, 0]
    
    width, height = env.MAZE_WIDTH, env.MAZE_HEIGHT
    dist = [[-1] * height for _ in range(width)]
    parent = [[None] * height for _ in range(width)]
    queue = []
    index = 0
    dist[player_x][player_y] = 0
    queue.append((player_x, player_y))
    
    while index < len(queue):
        x, y = queue[index]
        index += 1
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and env.maze[nx, ny] == 0 and dist[nx][ny] == -1:
                dist[nx][ny] = dist[x][y] + 1
                parent[nx][ny] = (x, y)
                queue.append((nx, ny))
    
    min_dist = float('inf')
    target_fruit = None
    for fruit in env.fruits:
        d = dist[fruit[0]][fruit[1]]
        if d != -1 and d < min_dist:
            min_dist = d
            target_fruit = fruit
    
    if target_fruit is None:
        return [0, 0, 0]
    
    current = target_fruit
    while parent[current[0]][current[1]] != (player_x, player_y):
        current = parent[current[0]][current[1]]
    
    dx = current[0] - player_x
    dy = current[1] - player_y
    
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