def policy(env):
    # Strategy: Prioritize placing crystals that guide laser toward target by reflecting or splitting the beam.
    # If cursor is on a valid laser path segment, place the best crystal type (blue for reflection, green/purple for splitting).
    # Otherwise, move toward the closest laser path to setup future placements. Avoid invalid placements and oscillation.
    import math
    
    if env.game_over:
        return [0, 0, 0]
    
    CELL_SIZE = env.CELL_SIZE
    cursor_world = [env.cursor_pos[0] * CELL_SIZE + CELL_SIZE/2, env.cursor_pos[1] * CELL_SIZE + CELL_SIZE/2]
    
    # Check if cursor is on any laser path segment
    on_path = False
    incoming_dir = None
    for start, end in env.laser_paths:
        seg_vec = [end[0]-start[0], end[1]-start[1]]
        seg_len = math.sqrt(seg_vec[0]**2 + seg_vec[1]**2)
        if seg_len < 1e-6:
            continue
        t = max(0, min(1, ((cursor_world[0]-start[0])*seg_vec[0] + (cursor_world[1]-start[1])*seg_vec[1]) / (seg_len**2)))
        proj = [start[0] + t*seg_vec[0], start[1] + t*seg_vec[1]]
        dist = math.sqrt((proj[0]-cursor_world[0])**2 + (proj[1]-cursor_world[1])**2)
        if dist < CELL_SIZE/4:
            on_path = True
            incoming_dir = [seg_vec[0]/seg_len, seg_vec[1]/seg_len]
            break
    
    # If on path and valid placement spot, place or cycle to best crystal
    if on_path:
        valid_placement = True
        if math.sqrt((cursor_world[0]-env.target_pos[0])**2 + (cursor_world[1]-env.target_pos[1])**2) < CELL_SIZE:
            valid_placement = False
        if math.sqrt((cursor_world[0]-env.laser_source[0])**2 + (cursor_world[1]-env.laser_source[1])**2) < CELL_SIZE:
            valid_placement = False
        for c_pos, _ in env.crystals:
            c_world = [c_pos[0]*CELL_SIZE+CELL_SIZE/2, c_pos[1]*CELL_SIZE+CELL_SIZE/2]
            if math.sqrt((c_world[0]-cursor_world[0])**2 + (c_world[1]-cursor_world[1])**2) < CELL_SIZE:
                valid_placement = False
                break
        
        if valid_placement:
            target_dir = [env.target_pos[0]-cursor_world[0], env.target_pos[1]-cursor_world[1]]
            target_len = math.sqrt(target_dir[0]**2 + target_dir[1]**2)
            if target_len > 1e-6:
                target_dir = [target_dir[0]/target_len, target_dir[1]/target_len]
                # Compute outgoing directions for each crystal type
                dots = []
                # Blue reflector
                blue_out = [-incoming_dir[1], -incoming_dir[0]]
                blue_dot = blue_out[0]*target_dir[0] + blue_out[1]*target_dir[1]
                dots.append(blue_dot)
                # Green rotator
                green_out = [incoming_dir[1], -incoming_dir[0]]
                green_dot = green_out[0]*target_dir[0] + green_out[1]*target_dir[1]
                dots.append(green_dot)
                # Purple splitter (use green-like output)
                purple_out = [incoming_dir[1], -incoming_dir[0]]
                purple_dot = purple_out[0]*target_dir[0] + purple_out[1]*target_dir[1]
                dots.append(purple_dot)
                
                best_type = dots.index(max(dots))
                if env.selected_crystal == best_type:
                    return [0, 1, 0]  # Place crystal
                else:
                    return [0, 0, 1]  # Cycle to best type
            else:
                return [0, 1, 0]  # Place if already at target
    
    # Not on path: move toward closest laser segment
    min_dist = float('inf')
    target_grid = [env.GRID_COLS//2, env.GRID_ROWS//2]  # Default to center
    for start, end in env.laser_paths:
        seg_vec = [end[0]-start[0], end[1]-start[1]]
        seg_len = math.sqrt(seg_vec[0]**2 + seg_vec[1]**2)
        if seg_len < 1e-6:
            continue
        t = max(0, min(1, ((cursor_world[0]-start[0])*seg_vec[0] + (cursor_world[1]-start[1])*seg_vec[1]) / (seg_len**2)))
        proj = [start[0] + t*seg_vec[0], start[1] + t*seg_vec[1]]
        dist = math.sqrt((proj[0]-cursor_world[0])**2 + (proj[1]-cursor_world[1])**2)
        if dist < min_dist:
            min_dist = dist
            target_grid = [int(proj[0] / CELL_SIZE), int(proj[1] / CELL_SIZE)]
    
    # Move toward target grid cell
    dx = target_grid[0] - env.cursor_pos[0]
    dy = target_grid[1] - env.cursor_pos[1]
    if abs(dx) > abs(dy):
        return [4 if dx > 0 else 3, 0, 0]
    elif dy != 0:
        return [2 if dy > 0 else 1, 0, 0]
    else:
        return [0, 0, 0]  # No movement if already on target