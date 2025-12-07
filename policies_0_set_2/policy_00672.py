def policy(env):
    # Strategy: Prioritize clearing stopped vehicles at intersections to minimize penalties and maximize flow.
    # For each intersection, compute urgency based on stopped vehicles and approaching traffic to set optimal light phase.
    # Avoid unnecessary toggles by checking current state and last action to prevent no-ops.
    
    n_intersections = len(env.intersection_coords)
    STOPPED_DIST_SQ = 5625  # 75^2 pixels
    APPROACH_DIST_SQ = 40000  # 200^2 pixels
    best_score = -1
    best_idx = -1
    best_flow = None
    
    for idx, pos in enumerate(env.intersection_coords):
        x, y = pos
        v_stopped, h_stopped = 0, 0
        v_approaching, h_approaching = 0, 0
        
        for v in env.vehicles:
            if not v['alive'] or pos in v['crossed_intersections']:
                continue
            dx = v['pos'][0] - x
            dy = v['pos'][1] - y
            dist_sq = dx*dx + dy*dy
            
            if dist_sq < STOPPED_DIST_SQ and v['state'] == 'stopped':
                if v['direction'] == 'vertical':
                    v_stopped += 1
                else:
                    h_stopped += 1
            if dist_sq < APPROACH_DIST_SQ:
                if v['direction'] == 'vertical':
                    v_approaching += 1
                else:
                    h_approaching += 1
        
        if v_stopped + h_stopped > 0:
            score = abs(v_stopped - h_stopped)
            desired_flow = 'vertical' if v_stopped > h_stopped else 'horizontal'
        else:
            score = abs(v_approaching - h_approaching)
            desired_flow = 'vertical' if v_approaching > h_approaching else 'horizontal'
        
        if score > best_score:
            best_score = score
            best_idx = idx
            best_flow = desired_flow
    
    if best_score <= 0:
        return [0, 0, 0]
    
    current_idx = env.selected_intersection_idx
    if current_idx != best_idx:
        n = n_intersections
        clockwise = (best_idx - current_idx) % n
        counter = (current_idx - best_idx) % n
        if clockwise <= counter:
            return [1, 0, 0]
        else:
            return [2, 0, 0]
    else:
        current_flow = env.intersections[env.intersection_coords[current_idx]]['flow']
        if current_flow == best_flow:
            return [0, 0, 0]
        elif best_flow == 'vertical':
            if env.last_action[1] == 0:
                return [0, 1, 0]
            else:
                return [0, 0, 0]
        else:
            if env.last_action[2] == 0:
                return [0, 0, 1]
            else:
                return [0, 0, 0]