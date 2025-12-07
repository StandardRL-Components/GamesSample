def policy(env):
    """
    Greedy box-pushing strategy: For the selected box, push towards its target if possible.
    If blocked or already on target, select the next unplaced box. Avoids invalid pushes and
    prioritizes moves that directly reduce Manhattan distance to target.
    """
    if env.game_over:
        return [0, 0, 0]
    
    current_idx = env.selected_box_idx
    current_pos = env.boxes[current_idx]
    target_pos = env.targets[current_idx]
    
    # If current box isn't on target, try to push it toward target
    if current_pos != target_pos:
        best_dir = None
        min_dist = float('inf')
        dir_map = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        
        for direction, (dx, dy) in dir_map.items():
            new_pos = (current_pos[0] + dx, current_pos[1] + dy)
            # Check if push is valid (within bounds and no collision)
            if (0 <= new_pos[0] < env.grid_size and 0 <= new_pos[1] < env.grid_size and
                not any(new_pos == box for i, box in enumerate(env.boxes) if i != current_idx)):
                new_dist = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])
                if new_dist < min_dist:
                    min_dist = new_dist
                    best_dir = direction
        
        if best_dir is not None:
            return [best_dir, 1, 0]
    
    # If current box is on target or can't be pushed, select next unplaced box
    unplaced = [i for i in range(env.num_boxes) if env.boxes[i] != env.targets[i]]
    if unplaced:
        current = env.selected_box_idx
        # Find closest unplaced box in cyclic order
        for delta in [1, -1, 2, -2]:
            new_idx = (current + delta) % env.num_boxes
            if new_idx in unplaced:
                action_map = {1: 1, -1: 3, 2: 2, -2: 4}
                return [action_map[delta], 0, 0]
    
    return [0, 0, 0]