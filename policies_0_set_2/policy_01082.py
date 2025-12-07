def policy(env):
    # This policy maximizes matches by evaluating all adjacent swaps for immediate matches.
    # It prioritizes swaps that create the longest matches (3+ tiles) to clear more tiles per move.
    # If no match is found, it selects a tile to enable future matches, moving the cursor efficiently.
    
    def find_matches(grid):
        matches = set()
        # Horizontal matches
        for r in range(env.GRID_HEIGHT):
            for c in range(env.GRID_WIDTH - 2):
                if grid[r, c] != -1 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
                    if c + 3 < env.GRID_WIDTH and grid[r, c] == grid[r, c+3]:
                        matches.add((r, c+3))
                        if c + 4 < env.GRID_WIDTH and grid[r, c] == grid[r, c+4]:
                            matches.add((r, c+4))
        # Vertical matches
        for c in range(env.GRID_WIDTH):
            for r in range(env.GRID_HEIGHT - 2):
                if grid[r, c] != -1 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
                    if r + 3 < env.GRID_HEIGHT and grid[r, c] == grid[r+3, c]:
                        matches.add((r+3, c))
                        if r + 4 < env.GRID_HEIGHT and grid[r, c] == grid[r+4, c]:
                            matches.add((r+4, c))
        return matches

    if env.selected_pos is not None:
        sel_r, sel_c = env.selected_pos
        cur_r, cur_c = env.cursor_pos
        if abs(sel_r - cur_r) + abs(sel_c - cur_c) == 1:
            return [0, 1, 0]  # Swap if adjacent
        else:
            # Move toward selected tile's adjacent positions
            adjacents = []
            if sel_r > 0: adjacents.append((sel_r-1, sel_c, 1))  # Up
            if sel_r < env.GRID_HEIGHT-1: adjacents.append((sel_r+1, sel_c, 2))  # Down
            if sel_c > 0: adjacents.append((sel_r, sel_c-1, 3))  # Left
            if sel_c < env.GRID_WIDTH-1: adjacents.append((sel_r, sel_c+1, 4))  # Right
            
            # Prioritize moves that lead to matches
            best_move = None
            best_score = -1
            for r, c, move in adjacents:
                temp_grid = env.grid.copy()
                temp_grid[sel_r, sel_c], temp_grid[r, c] = temp_grid[r, c], temp_grid[sel_r, sel_c]
                match_count = len(find_matches(temp_grid))
                if match_count > best_score:
                    best_score = match_count
                    best_move = move
            if best_move is not None:
                return [best_move, 0, 0]
            return [1, 0, 0]  # Default to up if no match found

    else:
        # Find best swap opportunity
        best_score = -1
        best_tile = None
        for r in range(env.GRID_HEIGHT):
            for c in range(env.GRID_WIDTH):
                # Check right swap
                if c < env.GRID_WIDTH - 1:
                    temp_grid = env.grid.copy()
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    score = len(find_matches(temp_grid))
                    if score > best_score:
                        best_score = score
                        best_tile = (r, c)
                # Check down swap
                if r < env.GRID_HEIGHT - 1:
                    temp_grid = env.grid.copy()
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    score = len(find_matches(temp_grid))
                    if score > best_score:
                        best_score = score
                        best_tile = (r, c)
        
        if best_tile is None:
            best_tile = (env.GRID_HEIGHT // 2, env.GRID_WIDTH // 2)
        
        t_r, t_c = best_tile
        c_r, c_c = env.cursor_pos
        if c_r == t_r and c_c == t_c:
            return [0, 1, 0]  # Select tile
        elif c_c < t_c:
            return [4, 0, 0]  # Move right
        elif c_c > t_c:
            return [3, 0, 0]  # Move left
        elif c_r < t_r:
            return [2, 0, 0]  # Move down
        else:
            return [1, 0, 0]  # Move up