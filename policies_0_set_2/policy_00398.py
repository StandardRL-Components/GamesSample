def policy(env):
    # Strategy: Press space (a1=1) when a note is near the beat line to maximize score from hits while avoiding misses.
    # Check a 5x11 pixel region around each lane's beat line position for note colors (with tolerance for anti-aliasing).
    obs = env.render()
    tol = 50  # Color matching tolerance
    y_range = range(150, 161)  # y-coordinates near beat line (depth 150 → y=155)
    note_colors = env.COLOR_NOTE_LANES
    num_lanes = env.NUM_LANES
    iso_origin_x = env.ISO_ORIGIN_X
    lane_width = env.LANE_WIDTH
    
    # Compute x-center for each lane at beat line depth
    for lane in range(num_lanes):
        x_center = int(round(iso_origin_x + (lane - (num_lanes-1)/2.0) * lane_width * 0.707))
        target_color = note_colors[lane]
        # Check 5x11 region around lane center
        for y in y_range:
            for dx in (-2, -1, 0, 1, 2):
                x = x_center + dx
                if 0 <= x < env.SCREEN_WIDTH and 0 <= y < env.SCREEN_HEIGHT:
                    pixel = obs[y, x]
                    if all(abs(pixel[i] - target_color[i]) <= tol for i in range(3)):
                        return [0, 1, 0]  # Press space when note detected
    return [0, 0, 0]  # No action otherwise