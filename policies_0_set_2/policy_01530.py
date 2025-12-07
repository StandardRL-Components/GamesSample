def policy(env):
    # Strategy: Optimize rhythm game performance by precisely timing hits when notes align with target line.
    # Use RGB observation to detect note positions and only hit when notes are within the perfect hit window.
    # Focus on minimizing misses by being conservative with hits while maximizing combo potential.
    if hasattr(env, 'game_over') and env.game_over:
        return [0, 0, 0]
    
    obs = env._get_observation()
    target_y = 350  # Target line y-position
    hit_window = 20  # Reduced hit window for precision (~3*6.67 pixels)
    lane_width = 160  # 640/4 lanes
    note_colors = [(50, 150, 255), (50, 255, 150), (255, 200, 50), (255, 80, 80)]
    movement_map = [3, 2, 1, 4]  # Map lanes to movements: left, down, up, right
    
    best_lane = -1
    best_score = 0
    
    for lane in range(4):
        color = note_colors[lane]
        score = 0
        # Check precise hit window around target line
        for y in range(target_y - hit_window//2, target_y + hit_window//2):
            for x in range(lane*lane_width, (lane+1)*lane_width):
                r, g, b = obs[y, x]
                # Color matching with tolerance
                if (abs(r - color[0]) < 50 and abs(g - color[1]) < 50 and abs(b - color[2]) < 50):
                    score += 1
        if score > best_score:
            best_score = score
            best_lane = lane
    
    # Only hit if strong note presence detected in hit window
    if best_lane >= 0 and best_score > 30:
        return [movement_map[best_lane], 1, 0]
    return [0, 0, 0]