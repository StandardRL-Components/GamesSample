def policy(env):
    # Strategy: Prioritize detonating when mines cover rocks for immediate reward, then place mines on cells with maximum rock coverage to set up future detonations. Move efficiently towards optimal placement cells.
    import math
    
    # Helper function to count rocks in 3x3 area around a cell
    def count_rocks_around(x, y):
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.GRID_W and 0 <= ny < env.GRID_H and env.grid[nx, ny] == env.ROCK:
                    count += 1
        return count

    # Check if detonation is possible and beneficial
    mines = []
    for x in range(env.GRID_W):
        for y in range(env.GRID_H):
            if env.grid[x, y] == env.MINE:
                mines.append((x, y))
    
    if mines and env.prev_action[2] == 0:  # Check shift not held
        # Verify at least one mine covers a rock
        for x, y in mines:
            if count_rocks_around(x, y) > 0:
                return [0, 0, 1]  # Detonate

    # If we can place a mine and current cell is empty with rocks nearby
    cx, cy = env.cursor_pos
    if (env.moves_remaining > 0 and env.grid[cx, cy] == env.EMPTY and 
        count_rocks_around(cx, cy) > 0 and env.prev_action[1] == 0):
        return [0, 1, 0]  # Place mine

    # Find best placement cell (empty with max rocks in 3x3)
    best_score = -1
    best_cell = None
    for x in range(env.GRID_W):
        for y in range(env.GRID_H):
            if env.grid[x, y] == env.EMPTY:
                score = count_rocks_around(x, y)
                if score > best_score:
                    best_score = score
                    best_cell = (x, y)
    
    # Move towards best cell if found
    if best_cell:
        tx, ty = best_cell
        dx = tx - cx
        dy = ty - cy
        
        if dx != 0:
            return [4 if dx > 0 else 3, 0, 0]
        if dy != 0:
            return [2 if dy > 0 else 1, 0, 0]
    
    return [0, 0, 0]  # Default no-op