def policy(env):
    # Strategy: Maximize immediate matches by evaluating all possible swaps from current cursor position.
    # If selected, swap only if it creates a match to avoid penalties. Otherwise cancel selection.
    # In neutral state, move cursor to gems with highest match potential using efficient pathfinding.
    # Prioritize horizontal matches first due to higher cascade potential in this environment.
    
    if env.game_over or len(env.animations) > 0:
        return [0, 0, 0]
    
    grid = env.grid
    cursor = env.cursor_pos
    state = env.selection_state
    
    def find_matches(grid):
        matches = set()
        for r in range(8):
            for c in range(8):
                if grid[r, c] == -1:
                    continue
                # Horizontal
                if c < 6 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(c, r), (c+1, r), (c+2, r)])
                # Vertical
                if r < 6 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(c, r), (c, r+1), (c, r+2)])
        return matches
    
    if state == "SELECTED":
        selected = env.selected_gem_coord
        best_dir = 0
        best_matches = 0
        for dir, (dx, dy) in enumerate([(0,-1), (0,1), (-1,0), (1,0)], 1):
            nx, ny = selected[0] + dx, selected[1] + dy
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue
            test_grid = grid.copy()
            test_grid[selected[1], selected[0]], test_grid[ny, nx] = test_grid[ny, nx], test_grid[selected[1], selected[0]]
            matches = find_matches(test_grid)
            if len(matches) > best_matches:
                best_matches = len(matches)
                best_dir = dir
        if best_matches > 0:
            return [best_dir, 0, 0]
        else:
            return [0, 1, 0]
    
    else:  # NEUTRAL state
        best_score = -1
        best_move = [0, 0, 0]
        cx, cy = cursor
        
        # Check current position for potential matches
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue
            test_grid = grid.copy()
            test_grid[cy, cx], test_grid[ny, nx] = test_grid[ny, nx], test_grid[cy, cx]
            matches = find_matches(test_grid)
            if len(matches) > best_score:
                best_score = len(matches)
                best_move = [0, 1, 0]  # Select current gem
        
        # If no good swap at current position, move to best adjacent position
        if best_score <= 0:
            for dx, dy, move in [(0,-1,1), (0,1,2), (-1,0,3), (1,0,4)]:
                nx, ny = (cx + dx) % 8, (cy + dy) % 8
                test_grid = grid.copy()
                test_grid[cy, cx], test_grid[ny, nx] = test_grid[ny, nx], test_grid[cy, cx]
                matches = find_matches(test_grid)
                if len(matches) > best_score:
                    best_score = len(matches)
                    best_move = [move, 0, 0]
        
        return best_move