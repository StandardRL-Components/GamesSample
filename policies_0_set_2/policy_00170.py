def policy(env):
    # Strategy: Prioritize making matches by scanning for valid swaps. If a gem is selected, move to an adjacent gem that creates a match. If no gem is selected, find any valid swap and move to select the first gem. If no swaps exist, move the cursor in a snake pattern to explore the board. Avoid using shift to preserve moves.
    if env.game_over:
        return [0, 0, 0]
    
    def has_match(grid):
        for r in range(10):
            for c in range(8):
                if grid[r, c] != 0 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    return True
        for c in range(10):
            for r in range(8):
                if grid[r, c] != 0 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    return True
        return False

    def swap_causes_match(x1, y1, x2, y2):
        grid_copy = env.grid.copy()
        grid_copy[y1, x1], grid_copy[y2, x2] = grid_copy[y2, x2], grid_copy[y1, x1]
        return has_match(grid_copy)

    if env.selected_gem is not None:
        sx, sy = env.selected_gem
        adjacent = [(sx+1, sy), (sx-1, sy), (sx, sy+1), (sx, sy-1)]
        for nx, ny in adjacent:
            if 0 <= nx < 10 and 0 <= ny < 10:
                if swap_causes_match(sx, sy, nx, ny):
                    cx, cy = env.cursor_pos
                    if cx == nx and cy == ny:
                        return [0, 1, 0]
                    if cx < nx:
                        return [4, 0, 0]
                    if cx > nx:
                        return [3, 0, 0]
                    if cy < ny:
                        return [2, 0, 0]
                    if cy > ny:
                        return [1, 0, 0]
        cx, cy = env.cursor_pos
        if cx == sx and cy == sy:
            return [0, 1, 0]
        if cx < sx:
            return [4, 0, 0]
        if cx > sx:
            return [3, 0, 0]
        if cy < sy:
            return [2, 0, 0]
        return [1, 0, 0]
    else:
        for y in range(10):
            for x in range(10):
                if x < 9 and swap_causes_match(x, y, x+1, y):
                    cx, cy = env.cursor_pos
                    if cx == x and cy == y:
                        return [0, 1, 0]
                    if cx < x:
                        return [4, 0, 0]
                    if cx > x:
                        return [3, 0, 0]
                    if cy < y:
                        return [2, 0, 0]
                    if cy > y:
                        return [1, 0, 0]
                if y < 9 and swap_causes_match(x, y, x, y+1):
                    cx, cy = env.cursor_pos
                    if cx == x and cy == y:
                        return [0, 1, 0]
                    if cx < x:
                        return [4, 0, 0]
                    if cx > x:
                        return [3, 0, 0]
                    if cy < y:
                        return [2, 0, 0]
                    if cy > y:
                        return [1, 0, 0]
        cx, cy = env.cursor_pos
        if cy < 9:
            if cy % 2 == 0:
                if cx < 9:
                    return [4, 0, 0]
                return [2, 0, 0]
            else:
                if cx > 0:
                    return [3, 0, 0]
                return [2, 0, 0]
        else:
            if cx > 0:
                return [3, 0, 0]
            return [1, 0, 0]