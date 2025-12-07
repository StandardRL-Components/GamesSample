def policy(env):
    # Strategy: Move towards nearest fruit while avoiding bombs, using attraction-repulsion vector.
    # Prioritizes slicing fruits (reward) and evading bombs (penalty) by calculating desired direction
    # based on weighted distances to objects, then selects discrete movement action closest to that direction.
    current_x, current_y = env.slicer_pos.x, env.slicer_pos.y
    att_x, att_y = 0.0, 0.0
    rep_x, rep_y = 0.0, 0.0
    
    for obj in env.objects:
        dx = obj['pos'].x - current_x
        dy = obj['pos'].y - current_y
        dist = (dx*dx + dy*dy) ** 0.5
        if dist < 1.0:
            dist = 1.0
        weight = 1.0 / dist
        
        if obj['type'] == 'fruit':
            urgency = 1.0 + (obj['pos'].y / env.SCREEN_HEIGHT)
            att_x += dx * weight * urgency
            att_y += dy * weight * urgency
        else:
            rep_x -= dx * weight
            rep_y -= dy * weight
    
    desired_x = att_x + rep_x
    desired_y = att_y + rep_y
    directions = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]
    best_index = 0
    best_dot = -10**9
    
    for idx, (dx, dy) in enumerate(directions):
        dot = desired_x * dx + desired_y * dy
        if dot > best_dot:
            best_dot = dot
            best_index = idx
            
    return [best_index, 0, 0]