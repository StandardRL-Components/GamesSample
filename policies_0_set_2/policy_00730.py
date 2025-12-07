def policy(env):
    # Strategy: Prioritize pushing crystals toward targets by evaluating each movement direction's impact on total crystal-target Manhattan distance.
    # Select the action that minimizes total distance, with tie-breaking favoring directions that don't push crystals away from targets.
    player_pos = env.player_pos
    crystals = env.crystals
    targets = env.targets
    grid = env.grid
    grid_w, grid_h = env.grid_w, env.grid_h
    
    best_action = [0, 0, 0]
    best_score = float('-inf')
    
    for move in [1, 2, 3, 4]:
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[move]
        new_x, new_y = player_pos[0] + dx, player_pos[1] + dy
        
        if not (0 <= new_x < grid_w and 0 <= new_y < grid_h) or grid[new_y][new_x] == '#':
            continue
            
        crystal_idx = None
        for i, c in enumerate(crystals):
            if c['pos'] == (new_x, new_y):
                crystal_idx = i
                break
                
        if crystal_idx is None:
            score = 0.0
        else:
            c_pos = crystals[crystal_idx]['pos']
            slide_x, slide_y = c_pos
            while True:
                next_x, next_y = slide_x + dx, slide_y + dy
                if not (0 <= next_x < grid_w and 0 <= next_y < grid_h) or grid[next_y][next_x] == '#':
                    break
                blocking_crystal = False
                for c2 in crystals:
                    if c2['pos'] == (next_x, next_y):
                        blocking_crystal = True
                        break
                if blocking_crystal:
                    break
                slide_x, slide_y = next_x, next_y
                
            if (slide_x, slide_y) == c_pos:
                continue
                
            old_dist = abs(c_pos[0] - targets[crystal_idx][0]) + abs(c_pos[1] - targets[crystal_idx][1])
            new_dist = abs(slide_x - targets[crystal_idx][0]) + abs(slide_y - targets[crystal_idx][1])
            score = old_dist - new_dist
            
        if score > best_score or (score == best_score and move < best_action[0]):
            best_score = score
            best_action = [move, 0, 0]
            
    return best_action