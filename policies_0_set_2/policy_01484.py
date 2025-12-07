def policy(env):
    # Strategy: Maximize tower defense by placing long-range lasers early to cover path and cannons near base for high damage.
    # Prioritize spots closest to base for placement, using distance squared to avoid sqrt. Move cursor to best empty spot.
    base_center = (env.SCREEN_WIDTH - 20, 100)
    spots = env.placement_spots
    occupied = env.occupied_placement_indices
    current_idx = env.cursor_index
    current_pos = spots[current_idx]
    
    # Find closest empty spot to base
    best_idx = current_idx
    min_dist_sq = float('inf')
    for idx, spot in enumerate(spots):
        if idx in occupied:
            continue
        dist_sq = (spot.x - base_center[0])**2 + (spot.y - base_center[1])**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_idx = idx
            
    # If at target spot, place tower (laser if left of center, else cannon)
    if current_idx == best_idx:
        return [0, int(spots[current_idx].x >= 320), int(spots[current_idx].x < 320)]
    
    # Move toward target spot using dominant axis
    target_pos = spots[best_idx]
    dx = target_pos.x - current_pos.x
    dy = target_pos.y - current_pos.y
    if abs(dx) > abs(dy):
        return [4 if dx > 0 else 3, 0, 0]
    else:
        return [2 if dy > 0 else 1, 0, 0]