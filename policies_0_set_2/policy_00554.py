def policy(env):
    """
    Maximizes reward by systematically scanning grid for optimal tower placements.
    Prioritizes building Cannons (cheaper) early and Archers (long-range) later.
    Only builds on path-adjacent empty cells when resources allow, ensuring coverage
    of enemy path while minimizing resource waste. Movement follows row-major pattern
    to efficiently explore all valid build locations.
    """
    x, y = env.cursor_pos
    # Check if current cell is buildable and adjacent to path
    if env.grid[x][y] == 0:
        adjacent = False
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.GRID_W and 0 <= ny < env.GRID_H and env.grid[nx][ny] == 1:
                adjacent = True
                break
        if adjacent:
            if env.resources >= 60 and env.current_wave > 5:
                return [0, 0, 1]  # Build Archer
            elif env.resources >= 40:
                return [0, 1, 0]  # Build Cannon

    # Move cursor in row-major order
    total_cells = env.GRID_W * env.GRID_H
    desired_index = env.steps % total_cells
    desired_x = desired_index % env.GRID_W
    desired_y = desired_index // env.GRID_W
    current_x, current_y = env.cursor_pos
    if current_x < desired_x:
        return [4, 0, 0]  # Right
    elif current_x > desired_x:
        return [3, 0, 0]  # Left
    elif current_y < desired_y:
        return [2, 0, 0]  # Down
    elif current_y > desired_y:
        return [1, 0, 0]  # Up
    else:
        return [0, 0, 0]  # Stay