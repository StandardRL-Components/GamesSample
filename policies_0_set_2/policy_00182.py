def policy(env):
    # Strategy: Avoid obstacles by moving to the safest lane (top/middle/bottom) based on obstacle positions in the danger zone (right of the snail).
    # This minimizes collision risk while progressing forward, maximizing reward by reaching the finish line.
    if env.game_over:
        return [0, 0, 0]
    
    snail_rect = env.snail_rect
    danger_zone_x_min = snail_rect.right
    danger_zone_x_max = snail_rect.right + 100
    lanes = [(200, 260), (260, 320), (320, 380)]
    lane_centers = [230, 290, 350]
    safe_lanes = [True, True, True]
    
    for obstacle in env.obstacles:
        obst_rect = obstacle['rect']
        if obst_rect.right > danger_zone_x_min and obst_rect.left < danger_zone_x_max:
            for i, (low, high) in enumerate(lanes):
                if not (obst_rect.bottom < low or obst_rect.top > high):
                    safe_lanes[i] = False
                    
    snail_center_y = snail_rect.y + snail_rect.height / 2
    dists = [abs(snail_center_y - center) for center in lane_centers]
    current_lane_index = min(range(3), key=lambda i: dists[i])
    
    if safe_lanes[current_lane_index]:
        return [0, 0, 0]
        
    safe_indices = [i for i, safe in enumerate(safe_lanes) if safe]
    if not safe_indices:
        return [0, 0, 0]
        
    best_lane_index = min(safe_indices, key=lambda i: abs(lane_centers[i] - snail_center_y))
    
    if best_lane_index < current_lane_index:
        return [1, 0, 0]
    else:
        return [2, 0, 0]