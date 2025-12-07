def policy(env):
    # Strategy: Place towers in optimal defensive grid positions to cover enemy descent paths.
    # Prioritize center and choke points first, then expand coverage. Build when resources allow.
    strategic_positions = [
        (320, 200),  # center
        (270, 200),  # left center
        (370, 200),  # right center
        (320, 150),  # center upper
        (320, 250),  # center lower
        (270, 150),  # left upper
        (370, 150),  # right upper
        (270, 250),  # left lower
        (370, 250)   # right lower
    ]
    
    # Find next available strategic position
    target_pos = None
    for pos in strategic_positions:
        occupied = any(abs(tower['pos'].x - pos[0]) < 5 and abs(tower['pos'].y - pos[1]) < 5 for tower in env.towers)
        if not occupied:
            target_pos = pos
            break
    
    # If all strategic positions occupied, maintain position
    if target_pos is None:
        target_pos = (env.cursor_pos.x, env.cursor_pos.y)
    
    # Move toward target position
    movement = 0
    cur_x, cur_y = env.cursor_pos.x, env.cursor_pos.y
    if abs(cur_x - target_pos[0]) > 5:
        movement = 4 if target_pos[0] > cur_x else 3
    elif abs(cur_y - target_pos[1]) > 5:
        movement = 2 if target_pos[1] > cur_y else 1
    
    # Build if at target with resources and capacity
    at_target = movement == 0
    can_build = (env.score >= env.TOWER_COST and 
                 len(env.towers) < env.MAX_TOWERS and
                 at_target)
    
    build = 1 if can_build else 0
    return [movement, build, 0]