def policy(env):
    # This policy maximizes immediate matches by evaluating all adjacent swaps for their match potential.
    # It prioritizes swaps that create the largest matches (>=3) and breaks ties by Manhattan distance to cursor.
    # If no immediate matches are found, it moves toward the closest potential swap to avoid no-ops.
    
    def count_matches(grid, pos1, pos2):
        # Simulate swap and count immediate matches without chain reactions
        r1, c1 = pos1
        r2, c2 = pos2
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return 0
        grid_copy = [row[:] for row in grid]
        grid_copy[r1][c1], grid_copy[r2][c2] = grid_copy[r2][c2], grid_copy[r1][c1]
        matches = 0
        # Check horizontal matches
        for r in range(env.GRID_HEIGHT):
            for c in range(env.GRID_WIDTH - 2):
                if grid_copy[r][c] != -1 and grid_copy[r][c] == grid_copy[r][c+1] == grid_copy[r][c+2]:
                    matches += 3
                    if c + 3 < env.GRID_WIDTH and grid_copy[r][c+3] == grid_copy[r][c]:
                        matches += 1
        # Check vertical matches
        for c in range(env.GRID_WIDTH):
            for r in range(env.GRID_HEIGHT - 2):
                if grid_copy[r][c] != -1 and grid_copy[r][c] == grid_copy[r+1][c] == grid_copy[r+2][c]:
                    matches += 3
                    if r + 3 < env.GRID_HEIGHT and grid_copy[r+3][c] == grid_copy[r][c]:
                        matches += 1
        return matches

    grid = [[env.grid[r, c] for c in range(env.GRID_WIDTH)] for r in range(env.GRID_HEIGHT)]
    cursor_r, cursor_c = env.cursor
    selections = env.selections

    if len(selections) == 1:
        sel_r, sel_c = selections[0]
        best_score = -1
        best_pos = None
        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = sel_r + dr, sel_c + dc
            if 0 <= nr < env.GRID_HEIGHT and 0 <= nc < env.GRID_WIDTH:
                score = count_matches(grid, (sel_r, sel_c), (nr, nc))
                if score > best_score:
                    best_score = score
                    best_pos = (nr, nc)
        if best_score > 0:
            if (cursor_r, cursor_c) == best_pos:
                return [0, 1, 0]  # Select adjacent gem
            else:
                dr = best_pos[0] - cursor_r
                dc = best_pos[1] - cursor_c
                if dr != 0:
                    return [2 if dr > 0 else 1, 0, 0]
                else:
                    return [4 if dc > 0 else 3, 0, 0]
        else:
            return [0, 0, 1]  # Clear selection if no good swap

    best_score = -1
    best_swap = None
    for r in range(env.GRID_HEIGHT):
        for c in range(env.GRID_WIDTH):
            if c < env.GRID_WIDTH - 1:
                score = count_matches(grid, (r, c), (r, c+1))
                if score > best_score:
                    best_score = score
                    best_swap = ((r, c), (r, c+1))
            if r < env.GRID_HEIGHT - 1:
                score = count_matches(grid, (r, c), (r+1, c))
                if score > best_score:
                    best_score = score
                    best_swap = ((r, c), (r+1, c))
    
    if best_score > 0:
        gem1, gem2 = best_swap
        if (cursor_r, cursor_c) in (gem1, gem2):
            return [0, 1, 0]  # Select current gem
        else:
            target = gem1 if abs(cursor_r - gem1[0]) + abs(cursor_c - gem1[1]) <= \
                abs(cursor_r - gem2[0]) + abs(cursor_c - gem2[1]) else gem2
            dr = target[0] - cursor_r
            dc = target[1] - cursor_c
            if dr != 0:
                return [2 if dr > 0 else 1, 0, 0]
            else:
                return [4 if dc > 0 else 3, 0, 0]
    else:
        # No matches found, move cursor to avoid no-op
        if cursor_c < env.GRID_WIDTH - 1:
            return [4, 0, 0]
        elif cursor_r < env.GRID_HEIGHT - 1:
            return [2, 0, 0]
        else:
            return [3, 0, 0]