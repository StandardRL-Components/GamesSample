def policy(env):
    # This policy uses a simplified Tetris strategy: prioritize clearing lines by minimizing height and holes.
    # It evaluates potential placements for the current piece by simulating drops for each rotation and horizontal position.
    # The best placement minimizes aggregate height and holes while maximizing line clears, using a weighted scoring.
    # Actions are chosen to rotate and move toward the optimal placement, then hard drop when aligned.
    
    import numpy as np
    
    def evaluate_placement(grid, piece_type, rotation, x):
        shape = env.PIECES[piece_type][rotation % len(env.PIECES[piece_type])]
        height = len(shape)
        width = len(shape[0])
        
        # Find drop position
        drop_y = 0
        for test_y in range(env.GRID_HEIGHT - height + 1):
            collides = False
            for dy in range(height):
                for dx in range(width):
                    if shape[dy][dx]:
                        grid_x = x + dx
                        grid_y = test_y + dy
                        if grid_y >= env.GRID_HEIGHT or (grid_y >= 0 and grid[grid_x, grid_y] != 0):
                            collides = True
                            break
                if collides:
                    break
            if collides:
                break
            drop_y = test_y
        if drop_y < 0:
            return -9999  # Invalid placement
        
        # Simulate grid after placement
        sim_grid = grid.copy()
        for dy in range(height):
            for dx in range(width):
                if shape[dy][dx]:
                    sim_grid[x + dx, drop_y + dy] = 1
        
        # Clear lines
        lines_cleared = 0
        for y in range(env.GRID_HEIGHT):
            if np.all(sim_grid[:, y] != 0):
                lines_cleared += 1
                sim_grid[:, 1:y+1] = sim_grid[:, 0:y]
                sim_grid[:, 0] = 0
        
        # Calculate features
        heights = [0] * env.GRID_WIDTH
        holes = 0
        for x in range(env.GRID_WIDTH):
            column = sim_grid[x, :]
            filled = np.where(column != 0)[0]
            if len(filled) > 0:
                heights[x] = filled[-1] + 1
                holes += np.sum(column[:heights[x]] == 0)
        aggregate_height = np.sum(heights)
        bumpiness = np.sum(np.abs(np.diff(heights)))
        
        # Weighted scoring (prioritize line clears, minimize height and holes)
        score = lines_cleared * 100 - aggregate_height * 0.5 - holes * 1.0 - bumpiness * 0.2
        return score
    
    current = env.current_piece
    grid = env.grid
    best_score = -99999
    best_rot = current['rotation']
    best_x = current['x']
    
    # Evaluate all possible placements
    num_rots = len(env.PIECES[current['type']])
    for rot in range(num_rots):
        shape = env.PIECES[current['type']][rot]
        width = len(shape[0])
        for x in range(env.GRID_WIDTH - width + 1):
            score = evaluate_placement(grid, current['type'], rot, x)
            if score > best_score:
                best_score = score
                best_rot = rot
                best_x = x
    
    # Determine action to reach best placement
    a0, a1, a2 = 0, 0, 0
    if current['rotation'] != best_rot:
        a1 = 1  # Rotate
    elif current['x'] < best_x:
        a0 = 4  # Right
    elif current['x'] > best_x:
        a0 = 3  # Left
    else:
        a2 = 1  # Hard drop
    
    return [a0, a1, a2]