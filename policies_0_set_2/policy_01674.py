def policy(env):
    # Strategy: Prioritize moves that maximize immediate cell fills by moving to unfilled tiles with the most same-color connections. Avoid moves that waste actions (penalized moves or no-ops). Break ties by direction order (up, down, left, right) for consistency.
    def simulate_fill(grid_colors, grid_filled, pos):
        x, y = pos
        if grid_filled[y, x]:
            return 0
        color = grid_colors[y, x]
        visited = set()
        queue = [(x, y)]
        visited.add((x, y))
        count = 0
        while queue:
            cx, cy = queue.pop(0)
            if not grid_filled[cy, cx]:
                count += 1
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 5 and 0 <= ny < 5 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    if grid_colors[ny, nx] == color:
                        queue.append((nx, ny))
        return count

    if env.game_over:
        return [0, 0, 0]
    
    current_pos = env.player_pos
    best_action = 0
    best_score = -1
    directions = [1, 2, 3, 4]
    for move in directions:
        dx, dy = 0, 0
        if move == 1: dy = -1
        elif move == 2: dy = 1
        elif move == 3: dx = -1
        elif move == 4: dx = 1
        new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
        if 0 <= new_x < 5 and 0 <= new_y < 5 and not env.grid_filled[new_y, new_x]:
            score = simulate_fill(env.grid_colors, env.grid_filled, (new_x, new_y))
            if score > best_score:
                best_score = score
                best_action = move
                
    return [best_action, 0, 0] if best_score > 0 else [0, 0, 0]