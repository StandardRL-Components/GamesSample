def policy(env):
    # Strategy: Prioritize immediate matches by simulating all shift moves at current cursor.
    # If no matches found, move cursor right to explore new positions for future matches.
    import numpy as np
    
    def find_matches(grid):
        matches = set()
        width, height = grid.shape
        for r in range(height):
            for c in range(width):
                if grid[c, r] == 0:
                    continue
                if c < width - 2 and grid[c, r] == grid[c+1, r] == grid[c+2, r]:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
                if r < height - 2 and grid[c, r] == grid[c, r+1] == grid[c, r+2]:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return list(matches)

    if env.game_over or env.moves_left <= 0:
        return [0, 0, 0]

    best_dir = 0
    best_matches = 0
    grid_copy = env.grid.copy()
    x, y = env.cursor

    for direction in [1, 2, 3, 4]:
        test_grid = grid_copy.copy()
        if direction in [1, 2]:
            col = test_grid[:, y].copy()
            roll_amt = -1 if direction == 1 else 1
            test_grid[:, y] = np.roll(col, roll_amt)
        else:
            row = test_grid[x, :].copy()
            roll_amt = -1 if direction == 3 else 1
            test_grid[x, :] = np.roll(row, roll_amt)
        
        matches = find_matches(test_grid)
        if len(matches) > best_matches:
            best_matches = len(matches)
            best_dir = direction

    if best_matches > 0:
        return [best_dir, 1, 0]

    return [4, 0, 0]