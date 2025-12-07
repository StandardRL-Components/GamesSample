def policy(env):
    # Strategy: Prioritize immediate matches by scanning all possible swaps, then navigate to best swap position.
    # If a gem is selected, evaluate adjacent swaps for matches; else find globally optimal swap and move toward it.
    # This maximizes score by focusing on highest-yield matches first within move constraints.
    
    if env.selected_gem_pos is not None:
        sel_x, sel_y = env.selected_gem_pos
        best_dir = 0
        best_matches = 0
        for direction in [1, 2, 3, 4]:
            dx, dy = (0, -1) if direction == 1 else (0, 1) if direction == 2 else (-1, 0) if direction == 3 else (1, 0)
            nx, ny = sel_x + dx, sel_y + dy
            if not (0 <= nx < env.GRID_WIDTH and 0 <= ny < env.GRID_HEIGHT):
                continue
            grid_copy = env.grid.copy()
            grid_copy[sel_y, sel_x], grid_copy[ny, nx] = grid_copy[ny, nx], grid_copy[sel_y, sel_x]
            matches = env._find_all_matches(grid_copy)
            if len(matches) > best_matches:
                best_matches = len(matches)
                best_dir = direction
        if best_matches > 0:
            return [best_dir, 0, 0]
        else:
            return [0, 0, 1]
    else:
        best_swap = None
        best_score = 0
        for y in range(env.GRID_HEIGHT):
            for x in range(env.GRID_WIDTH):
                for dx, dy in [(0,1), (1,0)]:
                    nx, ny = x + dx, y + dy
                    if nx >= env.GRID_WIDTH or ny >= env.GRID_HEIGHT:
                        continue
                    grid_copy = env.grid.copy()
                    grid_copy[y, x], grid_copy[ny, nx] = grid_copy[ny, nx], grid_copy[y, x]
                    matches = env._find_all_matches(grid_copy)
                    match_count = len(matches)
                    if match_count > best_score:
                        best_score = match_count
                        best_swap = (x, y)
        cur_x, cur_y = env.cursor_pos
        if best_swap is not None:
            t_x, t_y = best_swap
            if cur_x == t_x and cur_y == t_y:
                return [0, 1, 0]
            if abs(t_x - cur_x) > abs(t_y - cur_y):
                return [4 if t_x > cur_x else 3, 0, 0]
            else:
                return [2 if t_y > cur_y else 1, 0, 0]
        return [0, 0, 0]