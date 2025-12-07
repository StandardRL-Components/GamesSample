def policy(env):
    # Greedy one-step lookahead: Evaluates all push directions from current selection,
    # choosing the one that maximizes immediate correct pixel improvement. If no improvement
    # is possible, cycles selection to explore new positions without using moves.
    if env.game_over:
        return [0, 0, 0]
    
    n = env.GRID_DIM
    x, y = env.selected_pixel
    current_correct = 0
    for i in range(n):
        for j in range(n):
            if env.current_grid[i, j] == env.target_grid[i, j]:
                current_correct += 1
                
    best_improvement = -1
    best_direction = 0
    
    for direction in [1, 2, 3, 4]:
        grid_copy = env.current_grid.copy()
        if direction == 1:  # Up
            col = [grid_copy[i, x] for i in range(n)]
            new_col = col[1:] + [col[0]]
            for i in range(n):
                grid_copy[i, x] = new_col[i]
        elif direction == 2:  # Down
            col = [grid_copy[i, x] for i in range(n)]
            new_col = [col[-1]] + col[:-1]
            for i in range(n):
                grid_copy[i, x] = new_col[i]
        elif direction == 3:  # Left
            row = [grid_copy[y, j] for j in range(n)]
            new_row = row[1:] + [row[0]]
            for j in range(n):
                grid_copy[y, j] = new_row[j]
        elif direction == 4:  # Right
            row = [grid_copy[y, j] for j in range(n)]
            new_row = [row[-1]] + row[:-1]
            for j in range(n):
                grid_copy[y, j] = new_row[j]
                
        new_correct = 0
        for i in range(n):
            for j in range(n):
                if grid_copy[i, j] == env.target_grid[i, j]:
                    new_correct += 1
                    
        improvement = new_correct - current_correct
        if improvement > best_improvement:
            best_improvement = improvement
            best_direction = direction
            
    if best_improvement > 0:
        return [best_direction, 0, 0]
    else:
        return [0, 1, 0]