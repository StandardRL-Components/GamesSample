def policy(env):
    # Strategy: Prioritize immediate matches for reward, then move towards largest available match to maximize score.
    # Avoid penalties by only pressing space on valid matches and moving during animations.
    def bfs(grid, x, y):
        if grid[y][x] is None:
            return []
        color = grid[y][x]["color"]
        queue = [(x, y)]
        visited = set(queue)
        matches = []
        while queue:
            cx, cy = queue.pop(0)
            matches.append((cx, cy))
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 6 and 0 <= ny < 6 and (nx, ny) not in visited:
                    if grid[ny][nx] is not None and grid[ny][nx]["color"] == color:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return matches

    def manhattan(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    if env.game_phase != "IDLE":
        return [0, 0, 0]

    x, y = env.cursor_pos
    current_match = bfs(env.grid, x, y)
    if len(current_match) >= 3:
        return [0, 1, 0]

    best_size = 0
    best_pos = None
    for gy in range(6):
        for gx in range(6):
            if env.grid[gy][gx] is not None:
                match = bfs(env.grid, gx, gy)
                size = len(match)
                if size >= 3:
                    dist = manhattan(x, y, gx, gy)
                    if size > best_size or (size == best_size and dist < manhattan(x, y, best_pos[0], best_pos[1])):
                        best_size = size
                        best_pos = (gx, gy)

    if best_pos is not None:
        tx, ty = best_pos
        if tx > x:
            return [4, 0, 0]
        elif tx < x:
            return [3, 0, 0]
        elif ty > y:
            return [2, 0, 0]
        elif ty < y:
            return [1, 0, 0]
        return [0, 1, 0]

    move_cycle = (env.steps // 6) % 4
    moves = [4, 2, 3, 1]
    for i in range(4):
        action0 = moves[(move_cycle + i) % 4]
        if action0 == 4 and x < 5:
            break
        if action0 == 2 and y < 5:
            break
        if action0 == 3 and x > 0:
            break
        if action0 == 1 and y > 0:
            break
    else:
        action0 = 0
    return [action0, 0, 0]