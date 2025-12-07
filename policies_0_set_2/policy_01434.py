def policy(env):
    # Strategy: Use immediate reward simulation for each movement to select the action that minimizes box-target distances,
    # maximizes box placements, and avoids invalid moves. Prioritize moves that complete stages for maximum reward.
    if env.game_over:
        return [0, 0, 0]
    
    current_boxes = env.boxes
    current_targets = env.targets
    walls = env.walls
    grid_dims = env.grid_dims
    player_pos = env.player_pos
    
    # Precompute current total Manhattan distance and on-target count
    current_distance = 0
    unmatched_targets = list(current_targets)
    for box in current_boxes:
        closest_dist = float('inf')
        best_target = None
        for target in unmatched_targets:
            d = abs(box[0]-target[0]) + abs(box[1]-target[1])
            if d < closest_dist:
                closest_dist = d
                best_target = target
        if best_target is not None:
            current_distance += closest_dist
            unmatched_targets.remove(best_target)
    current_on_target = sum(1 for box in current_boxes if box in current_targets)
    
    best_reward = -float('inf')
    best_action = 0
    
    directions = [1, 2, 3, 4]
    for a0 in directions:
        reward = -10  # Default for invalid moves
        dr, dc = (0, -1) if a0 == 1 else (0, 1) if a0 == 2 else (-1, 0) if a0 == 3 else (1, 0)
        new_r, new_c = player_pos[0] + dr, player_pos[1] + dc
        
        # Check boundaries and walls
        if not (0 <= new_c < grid_dims[1] and 0 <= new_r < grid_dims[0]) or (new_r, new_c) in walls:
            continue
            
        # Check for box push
        box_idx = None
        for idx, box in enumerate(current_boxes):
            if box[0] == new_r and box[1] == new_c:
                box_idx = idx
                break
                
        if box_idx is not None:
            new_box_r, new_box_c = new_r + dr, new_c + dc
            if not (0 <= new_box_c < grid_dims[1] and 0 <= new_box_r < grid_dims[0]) or (new_box_r, new_box_c) in walls:
                continue
            if any(b[0] == new_box_r and b[1] == new_box_c for b in current_boxes):
                continue
                
            # Valid push: simulate new box positions
            new_boxes = [list(b) for b in current_boxes]
            new_boxes[box_idx] = [new_box_r, new_box_c]
            
            # Compute new total distance
            new_distance = 0
            unmatched_targets = list(current_targets)
            for box in new_boxes:
                closest_dist = float('inf')
                best_target = None
                for target in unmatched_targets:
                    d = abs(box[0]-target[0]) + abs(box[1]-target[1])
                    if d < closest_dist:
                        closest_dist = d
                        best_target = target
                if best_target is not None:
                    new_distance += closest_dist
                    unmatched_targets.remove(best_target)
                    
            # Compute on-target change
            new_on_target = sum(1 for box in new_boxes if box in current_targets)
            reward = (current_distance - new_distance) * 0.1 + (new_on_target - current_on_target) * 5
            
            # Check for stage completion
            if set(map(tuple, new_boxes)) == set(map(tuple, current_targets)):
                reward += 50 if env.current_stage < 3 else 100
        else:
            # Valid move without push
            reward = 0.1
            
        if reward > best_reward:
            best_reward = reward
            best_action = a0
            
    return [best_action, 0, 0]