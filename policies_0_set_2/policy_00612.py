def policy(env):
    # Strategy: Minesweeper requires safe exploration. Prioritize revealing cells with low adjacent mine probability.
    # Avoid known mines, flag suspected mines, and reveal safe cells to maximize flood fill rewards.
    # Move systematically to uncover unrevealed cells while minimizing risk based on adjacent numbers.
    
    # Extract cursor position from env (read-only)
    cx, cy = env.cursor_pos
    grid_w, grid_h = env.grid_w, env.grid_h
    
    # Parse observation to infer cell states (revealed, flagged, number)
    obs = env._get_observation()
    cell_states = [[None] * grid_h for _ in range(grid_w)]
    start_x, start_y = (640 - 360) // 2, (400 - 360) // 2  # Grid offset
    cell_size = 36
    
    for x in range(grid_w):
        for y in range(grid_h):
            px = start_x + x * cell_size + cell_size // 2
            py = start_y + y * cell_size + cell_size // 2
            color = obs[py, px]
            
            # Check if revealed (lighter background)
            if np.allclose(color, [80, 88, 96], atol=20):
                cell_states[x][y] = ('revealed', 0)
            # Check numbers by color (approximate matches)
            elif np.allclose(color, [50, 150, 255], atol=20):
                cell_states[x][y] = ('revealed', 1)
            elif np.allclose(color, [50, 200, 100], atol=20):
                cell_states[x][y] = ('revealed', 2)
            elif np.allclose(color, [255, 100, 100], atol=20):
                cell_states[x][y] = ('revealed', 3)
            elif np.allclose(color, [150, 50, 255], atol=20):
                cell_states[x][y] = ('revealed', 4)
            elif np.allclose(color, [255, 150, 50], atol=20):
                cell_states[x][y] = ('revealed', 5)
            elif np.allclose(color, [50, 200, 200], atol=20):
                cell_states[x][y] = ('revealed', 6)
            elif np.allclose(color, [200, 50, 200], atol=20):
                cell_states[x][y] = ('revealed', 7)
            elif np.allclose(color, [100, 100, 100], atol=20):
                cell_states[x][y] = ('revealed', 8)
            # Check flagged (yellow polygon)
            elif np.allclose(color, [240, 180, 0], atol=20):
                cell_states[x][y] = ('flagged', None)
            # Otherwise unrevealed
            else:
                cell_states[x][y] = ('unrevealed', None)
    
    # Helper to check adjacent cells
    def get_adjacent(x, y):
        adj = []
        for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                adj.append((nx, ny))
        return adj
    
    # Check if current cell is safe to reveal (unrevealed and not flagged)
    if cell_states[cx][cy][0] == 'unrevealed':
        return [0, 1, 0]  # Reveal
    
    # Check if current cell should be flagged (unrevealed adjacent to number equal to adjacent mines)
    if cell_states[cx][cy][0] == 'unrevealed':
        for ax, ay in get_adjacent(cx, cy):
            if cell_states[ax][ay][0] == 'revealed':
                num = cell_states[ax][ay][1]
                if num > 0:
                    adj_unrevealed = [1 for nx, ny in get_adjacent(ax, ay) 
                                    if cell_states[nx][ny][0] == 'unrevealed']
                    if sum(adj_unrevealed) == num:
                        return [0, 0, 1]  # Flag
    
    # Move to nearest unrevealed cell (row-major search)
    for y in range(grid_h):
        for x in range(grid_w):
            if cell_states[x][y][0] == 'unrevealed':
                # Determine movement direction
                if x > cx:
                    return [4, 0, 0]  # Right
                elif x < cx:
                    return [3, 0, 0]  # Left
                elif y > cy:
                    return [2, 0, 0]  # Down
                elif y < cy:
                    return [1, 0, 0]  # Up
    
    return [0, 0, 0]  # Default no-op