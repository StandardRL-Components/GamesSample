def policy(env):
    # Strategy: Prioritize moves that create immediate matches for maximum reward and cascade potential.
    # Evaluate all possible crystal swaps to adjacent empty cells, choose the one with highest match count.
    # If no matches, move toward the nearest crystal with adjacent empty space to enable future matches.
    # Use cursor movement and selection actions to execute the best swap step-by-step.
    def find_matches(grid):
        matches = set()
        h, w = len(grid), len(grid[0])
        for r in range(h):
            for c in range(w - 2):
                if grid[r][c] > 0 and grid[r][c] == grid[r][c+1] == grid[r][c+2]:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
        for c in range(w):
            for r in range(h - 2):
                if grid[r][c] > 0 and grid[r][c] == grid[r+1][c] == grid[r+2][c]:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return matches

    def move_toward(cur, target):
        cx, cy = cur
        tx, ty = target
        if cx < tx: return [4, 0, 0]
        if cx > tx: return [3, 0, 0]
        if cy < ty: return [2, 0, 0]
        if cy > ty: return [1, 0, 0]
        return [0, 0, 0]

    if env.game_phase != "INPUT":
        return [0, 0, 0]

    grid = env.grid.tolist()
    best_score, best_crystal, best_empty = -1, None, None
    for r in range(env.GRID_HEIGHT):
        for c in range(env.GRID_WIDTH):
            if grid[r][c] > 0:
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < env.GRID_HEIGHT and 0 <= nc < env.GRID_WIDTH and grid[nr][nc] == 0:
                        temp = [row[:] for row in grid]
                        temp[nr][nc] = temp[r][c]
                        temp[r][c] = 0
                        matches = find_matches(temp)
                        score = len(matches)
                        if score > best_score:
                            best_score, best_crystal, best_empty = score, (c, r), (nc, nr)

    if best_crystal is None:
        for r in range(env.GRID_HEIGHT):
            for c in range(env.GRID_WIDTH):
                if grid[r][c] > 0:
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = r + dy, c + dx
                        if 0 <= nr < env.GRID_HEIGHT and 0 <= nc < env.GRID_WIDTH and grid[nr][nc] == 0:
                            best_crystal, best_empty = (c, r), (nc, nr)
                            break
                    if best_crystal is not None:
                        break
            if best_crystal is not None:
                break
        if best_crystal is None:
            return [0, 0, 0]

    cx, cy = env.cursor_pos
    if env.selected_crystal is not None:
        sel_x, sel_y = env.selected_crystal
        if (sel_x, sel_y) == best_crystal:
            if (cx, cy) == best_empty:
                return [0, 1, 0]
            return move_toward((cx, cy), best_empty)
        return [0, 1, 0]
    else:
        if (cx, cy) == best_crystal:
            return [0, 1, 0]
        return move_toward((cx, cy), best_crystal)