def policy(env):
    # This policy uses a greedy approach to push boxes towards targets while avoiding deadlocks.
    # It evaluates all possible moves, prioritizing those that push boxes onto targets, then moves
    # that reduce box-to-target distances, and finally moves that position the player for future pushes.
    # Secondary actions are unused in this environment and set to 0.
    
    def manhattan_dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    best_action = [0, 0, 0]
    best_score = -float('inf')
    player = env.player_pos
    boxes = [tuple(b) for b in env.box_positions]
    walls = env.wall_positions
    targets = env.target_positions
    
    for move in range(5):  # Evaluate all movement actions
        if move == 0:  # No-op
            score = -1.0  # Prefer actions over no-op
        else:
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[move]
            new_pos = (player[0] + dx, player[1] + dy)
            
            if new_pos in walls:  # Invalid move
                continue
                
            if new_pos in boxes:  # Attempt to push box
                box_new_pos = (new_pos[0] + dx, new_pos[1] + dy)
                if box_new_pos in walls or box_new_pos in boxes:  # Invalid push
                    continue
                    
                # Calculate push score
                score = 0.0
                if box_new_pos in targets:  # Immediate reward
                    score += 100.0
                else:
                    # Minimize distance to nearest target
                    min_dist = min(manhattan_dist(box_new_pos, t) for t in targets)
                    score += 10.0 / (1 + min_dist)
                    
                # Penalize moving boxes off targets
                if new_pos in targets:
                    score -= 5.0
                    
                # Add small risk bonus for edge pushes
                if (box_new_pos[0] <= 1 or box_new_pos[0] >= env.GRID_COLS - 2 or
                    box_new_pos[1] <= 1 or box_new_pos[1] >= env.GRID_ROWS - 2):
                    score += 0.5
            else:  # Regular move
                # Score by proximity to pushable boxes not on targets
                score = 0.0
                for box in boxes:
                    if box not in targets:
                        dist = manhattan_dist(new_pos, box)
                        score += 1.0 / (1 + dist)
        
        if score > best_score:
            best_score = score
            best_action[0] = move
    
    return best_action