def policy(env):
    # Strategy: Prioritize moves that create immediate matches to maximize score and minimize penalties.
    # In SELECTING phase, move cursor to the best crystal (one with a push direction that creates the most matches).
    # In PUSHING phase, choose the direction that maximizes immediate matches for the selected crystal.
    # Use Manhattan distance for cursor movement to avoid oscillation and break ties consistently.

    def find_matches(grid):
        matches = set()
        for r in range(env.GRID_SIZE):
            for c in range(env.GRID_SIZE):
                color = grid[r, c]
                if color == 0:
                    continue
                if c + 2 < env.GRID_SIZE and grid[r, c+1] == color and grid[r, c+2] == color:
                    matches.update([(r, c+i) for i in range(3)])
                if r + 2 < env.GRID_SIZE and grid[r+1, c] == color and grid[r+2, c] == color:
                    matches.update([(r+i, c) for i in range(3)])
        return matches

    def simulate_push(grid, y, x, direction):
        dy, dx = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][direction]
        new_grid = grid.copy()
        line = []
        curr_y, curr_x = y, x
        while 0 <= curr_y < env.GRID_SIZE and 0 <= curr_x < env.GRID_SIZE and new_grid[curr_y, curr_x] != 0:
            line.append((curr_y, curr_x))
            curr_y += dy
            curr_x += dx
        if not (0 <= curr_y < env.GRID_SIZE and 0 <= curr_x < env.GRID_SIZE):
            return 0
        for i in range(len(line)-1, -1, -1):
            old_y, old_x = line[i]
            new_grid[old_y+dy, old_x+dx] = new_grid[old_y, old_x]
        new_grid[y, x] = 0
        return len(find_matches(new_grid))

    if env.game_phase == 'SELECTING':
        best_score, best_pos, best_dir = -1, None, None
        for y in range(env.GRID_SIZE):
            for x in range(env.GRID_SIZE):
                if env.grid[y, x] == 0:
                    continue
                for d in [1,2,3,4]:
                    score = simulate_push(env.grid, y, x, d)
                    if score > best_score:
                        best_score, best_pos, best_dir = score, (y, x), d
        if best_pos is not None and best_score > 0:
            cy, cx = env.cursor_pos
            ty, tx = best_pos
            if cy == ty and cx == tx:
                return [0, 1, 0]
            if cx != tx:
                return [4 if cx < tx else 3, 0, 0]
            return [2 if cy < ty else 1, 0, 0]
        non_zero = np.transpose(np.nonzero(env.grid))
        if len(non_zero) > 0:
            cy, cx = env.cursor_pos
            distances = [abs(cy-y)+abs(cx-x) for y,x in non_zero]
            nearest = non_zero[np.argmin(distances)]
            ty, tx = nearest
            if cx != tx:
                return [4 if cx < tx else 3, 0, 0]
            return [2 if cy < ty else 1, 0, 0]
        return [0, 0, 0]

    elif env.game_phase == 'PUSHING':
        y, x = env.selected_crystal_pos
        best_dir, best_score = None, -1
        for d in [1,2,3,4]:
            score = simulate_push(env.grid, y, x, d)
            if score > best_score:
                best_dir, best_score = d, score
        if best_score > 0:
            return [best_dir, 0, 0]
        return [0, 1, 0]

    return [0, 0, 0]