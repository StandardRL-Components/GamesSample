def policy(env):
    # This policy maximizes score by first moving to and selecting a gem with a potential match, then swapping with an adjacent gem that creates the largest match.
    # It prioritizes immediate matches to maximize reward and uses deterministic movement to avoid oscillation.
    if env.game_state != 'AWAITING_INPUT':
        return [0, 0, 0]
    
    def check_match(grid, r, c):
        gem_type = grid[r, c]
        if gem_type == -1:
            return False
        # Check horizontal
        left = 0
        i = c - 1
        while i >= 0 and grid[r, i] == gem_type:
            left += 1
            i -= 1
        right = 0
        i = c + 1
        while i < env.GRID_WIDTH and grid[r, i] == gem_type:
            right += 1
            i += 1
        if left + right + 1 >= 3:
            return True
        # Check vertical
        up = 0
        i = r - 1
        while i >= 0 and grid[i, c] == gem_type:
            up += 1
            i -= 1
        down = 0
        i = r + 1
        while i < env.GRID_HEIGHT and grid[i, c] == gem_type:
            down += 1
            i += 1
        if up + down + 1 >= 3:
            return True
        return False

    cursor_r, cursor_c = env.cursor_pos[1], env.cursor_pos[0]
    if env.selected_pos is not None:
        sel_r, sel_c = env.selected_pos[1], env.selected_pos[0]
        if cursor_r != sel_r or cursor_c != sel_c:
            if cursor_c < sel_c:
                return [4, 0, 0]
            elif cursor_c > sel_c:
                return [3, 0, 0]
            elif cursor_r < sel_r:
                return [2, 0, 0]
            else:
                return [1, 0, 0]
        else:
            best_dir = None
            best_score = 0
            directions = [(1, -1, 0), (2, 1, 0), (3, 0, -1), (4, 0, 1)]
            for move, dr, dc in directions:
                nr, nc = sel_r + dr, sel_c + dc
                if 0 <= nr < env.GRID_HEIGHT and 0 <= nc < env.GRID_WIDTH:
                    sim_grid = env.grid.copy()
                    sim_grid[sel_r, sel_c], sim_grid[nr, nc] = sim_grid[nr, nc], sim_grid[sel_r, sel_c]
                    score = 0
                    if check_match(sim_grid, sel_r, sel_c):
                        score += 1
                    if check_match(sim_grid, nr, nc):
                        score += 1
                    if score > best_score:
                        best_score = score
                        best_dir = move
            if best_dir is not None:
                return [best_dir, 1, 0]
            else:
                return [0, 1, 0]
    else:
        best_target = None
        best_score = 0
        for r in range(env.GRID_HEIGHT):
            for c in range(env.GRID_WIDTH):
                if env.grid[r, c] == -1:
                    continue
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < env.GRID_HEIGHT and 0 <= nc < env.GRID_WIDTH and env.grid[nr, nc] != -1:
                        sim_grid = env.grid.copy()
                        sim_grid[r, c], sim_grid[nr, nc] = sim_grid[nr, nc], sim_grid[r, c]
                        score = 0
                        if check_match(sim_grid, r, c):
                            score += 1
                        if check_match(sim_grid, nr, nc):
                            score += 1
                        if score > best_score:
                            best_score = score
                            best_target = (c, r)
        if best_target is not None:
            t_c, t_r = best_target
            if cursor_c == t_c and cursor_r == t_r:
                return [0, 1, 0]
            else:
                if cursor_c < t_c:
                    return [4, 0, 0]
                elif cursor_c > t_c:
                    return [3, 0, 0]
                elif cursor_r < t_r:
                    return [2, 0, 0]
                else:
                    return [1, 0, 0]
        else:
            if cursor_c < env.GRID_WIDTH - 1:
                return [4, 0, 0]
            elif cursor_r < env.GRID_HEIGHT - 1:
                return [2, 0, 0]
            elif cursor_c > 0:
                return [3, 0, 0]
            else:
                return [1, 0, 0]