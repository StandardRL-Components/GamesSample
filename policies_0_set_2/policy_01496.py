def policy(env):
    # Strategy: Plant seeds in empty cells with the most adjacent empty plots to maximize spread potential.
    # Prioritize planting when possible, then move towards the best empty cell (highest empty neighbors).
    # Break ties by distance to current cursor to minimize movement and maximize turns for growth.
    grid = env.grid
    seeds = env.seeds
    cursor_x, cursor_y = env.cursor_pos
    current_empty = grid[cursor_y, cursor_x] == 0.0

    if seeds > 0 and current_empty:
        return [0, 1, 0]  # Plant immediately if possible

    if seeds > 0:
        best_score = -1
        best_pos = None
        for r in range(env.GRID_ROWS):
            for c in range(env.GRID_COLS):
                if grid[r, c] == 0.0:
                    empty_neighbors = 0
                    for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < env.GRID_ROWS and 0 <= nc < env.GRID_COLS and grid[nr, nc] == 0.0:
                            empty_neighbors += 1
                    dist = abs(r - cursor_y) + abs(c - cursor_x)
                    if empty_neighbors > best_score or (empty_neighbors == best_score and dist < best_dist):
                        best_score = empty_neighbors
                        best_dist = dist
                        best_pos = (r, c)

        if best_pos is not None:
            target_r, target_c = best_pos
            dx = target_c - cursor_x
            dy = target_r - cursor_y
            if abs(dx) > abs(dy):
                move = 4 if dx > 0 else 3
            else:
                move = 2 if dy > 0 else 1 if dy < 0 else 0
            return [move, 0, 0]

    return [0, 0, 0]  # Default no-op if no seeds or no empty cells