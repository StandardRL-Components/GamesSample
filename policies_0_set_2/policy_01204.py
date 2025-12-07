def policy(env):
    # Strategy: Move towards acorn while avoiding guards and vision cones. Prioritize safe moves that reduce Manhattan distance.
    # If no safe move exists, choose the non-wall move that minimizes distance. Use current guard positions for safety checks.
    TILE_SIZE = env.TILE_SIZE
    cone_length = 6 * TILE_SIZE
    cone_width = 3 * TILE_SIZE

    def is_point_in_triangle(pt, v1, v2, v3):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    current_pos = env.squirrel_pos
    acorn_pos = env.acorn_pos
    safe_actions = []
    
    for action in range(5):
        if action == 0:
            candidate = current_pos
        elif action == 1:
            candidate = (current_pos[0], current_pos[1] - 1)
        elif action == 2:
            candidate = (current_pos[0], current_pos[1] + 1)
        elif action == 3:
            candidate = (current_pos[0] - 1, current_pos[1])
        else:
            candidate = (current_pos[0] + 1, current_pos[1])
            
        if candidate in env.walls:
            continue
            
        safe = True
        candidate_pixel = ((candidate[0] + 0.5) * TILE_SIZE, (candidate[1] + 0.5) * TILE_SIZE)
        
        for guard in env.guards:
            guard_tile = (int(guard["pos"][0]), int(guard["pos"][1]))
            if candidate == guard_tile:
                safe = False
                break
                
            guard_pixel = (guard["pos"][0] * TILE_SIZE + TILE_SIZE/2, guard["pos"][1] * TILE_SIZE + TILE_SIZE/2)
            direction = (guard["direction"][0], guard["direction"][1])
            perp_dir = (-direction[1], direction[0])
            p2 = (guard_pixel[0] + direction[0]*cone_length + perp_dir[0]*cone_width,
                  guard_pixel[1] + direction[1]*cone_length + perp_dir[1]*cone_width)
            p3 = (guard_pixel[0] + direction[0]*cone_length - perp_dir[0]*cone_width,
                  guard_pixel[1] + direction[1]*cone_length - perp_dir[1]*cone_width)
                  
            if is_point_in_triangle(candidate_pixel, guard_pixel, p2, p3):
                safe = False
                break
                
        if safe:
            safe_actions.append((action, candidate))
            
    if safe_actions:
        best_action = None
        best_dist = float('inf')
        for action, pos in safe_actions:
            dist = abs(pos[0] - acorn_pos[0]) + abs(pos[1] - acorn_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_action = action
        return [best_action, 0, 0]
        
    best_action = 0
    best_dist = abs(current_pos[0] - acorn_pos[0]) + abs(current_pos[1] - acorn_pos[1])
    for action in [1,2,3,4]:
        if action == 1:
            candidate = (current_pos[0], current_pos[1] - 1)
        elif action == 2:
            candidate = (current_pos[0], current_pos[1] + 1)
        elif action == 3:
            candidate = (current_pos[0] - 1, current_pos[1])
        else:
            candidate = (current_pos[0] + 1, current_pos[1])
        if candidate not in env.walls:
            dist = abs(candidate[0] - acorn_pos[0]) + abs(candidate[1] - acorn_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_action = action
    return [best_action, 0, 0]