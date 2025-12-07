def policy(env):
    # Strategy: Extract maze structure from RGB observation to compute shortest path to exit using BFS.
    # This maximizes reward by efficiently navigating the maze to reach the exit before time runs out.
    info = env._get_info()
    difficulty = info['difficulty']
    w = min(25, 10 + difficulty * 2)
    h = min(25, 10 + difficulty * 2)
    padding = 40
    maze_render_width = 640 - 2 * padding
    maze_render_height = 400 - 2 * padding
    cell_w = maze_render_width / w
    cell_h = maze_render_height / h
    cell_size = min(cell_w, cell_h)
    total_maze_w = cell_size * w
    total_maze_h = cell_size * h
    offset_x = (640 - total_maze_w) / 2
    offset_y = (400 - total_maze_h) / 2
    obs = env._get_observation()
    
    walls = [[[True] * 4 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            x_center = int(offset_x + c * cell_size + cell_size // 2)
            y_center = int(offset_y + r * cell_size + cell_size // 2)
            for i, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
                sample_x = x_center + dx * int(cell_size // 2)
                sample_y = y_center + dy * int(cell_size // 2)
                if 0 <= sample_x < 640 and 0 <= sample_y < 400:
                    color = obs[sample_y, sample_x]
                    walls[r][c][i] = all(30 <= comp <= 50 for comp in color)
    
    player_pos = None
    exit_pos = None
    for r in range(h):
        for c in range(w):
            x0 = int(offset_x + c * cell_size)
            y0 = int(offset_y + r * cell_size)
            x1 = int(x0 + cell_size)
            y1 = int(y0 + cell_size)
            player_count = 0
            exit_count = 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if 0 <= y < 400 and 0 <= x < 640:
                        p = obs[y, x]
                        if abs(p[0] - 50) < 20 and abs(p[1] - 200) < 20 and abs(p[2] - 50) < 20:
                            player_count += 1
                        if abs(p[0] - 200) < 20 and abs(p[1] - 50) < 20 and abs(p[2] - 50) < 20:
                            exit_count += 1
            if player_count > 10:
                player_pos = (r, c)
            if exit_count > 10:
                exit_pos = (r, c)
    
    if player_pos is None or exit_pos is None:
        return [0, 0, 0]
    
    queue = [player_pos]
    visited = {player_pos: None}
    while queue:
        r, c = queue.pop(0)
        if (r, c) == exit_pos:
            break
        for dr, dc, move in [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                wall_idx = 0 if dr == -1 else 1 if dr == 1 else 2 if dc == -1 else 3
                if not walls[r][c][wall_idx]:
                    visited[(nr, nc)] = (r, c, move)
                    queue.append((nr, nc))
    
    path = []
    current = exit_pos
    while current != player_pos:
        if current not in visited:
            return [0, 0, 0]
        prev = visited[current]
        if prev is None:
            break
        r, c, move = prev
        path.append(move)
        current = (r, c)
    
    if path:
        return [path[-1], 0, 0]
    return [0, 0, 0]