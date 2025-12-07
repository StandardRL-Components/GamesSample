def policy(env):
    # Strategy: Tap columns when notes are within hit window (y=295-345) to maximize hits and minimize misses.
    # Prioritize center column (a1) and left/right (a0) based on note proximity to target line (y=320).
    # Break ties by choosing closest note to target, avoiding unnecessary taps when no notes are present.
    import numpy as np
    import pygame
    
    note_colors = [
        (245, 194, 231),  # Column 0 (left)
        (203, 166, 247),  # Column 1 (center)
        (148, 226, 213)   # Column 2 (right)
    ]
    grid_start_x = 200
    col_centers = [240, 320, 400]
    hit_window_ys = list(range(295, 346, 5))
    tolerance = 50
    
    arr = pygame.surfarray.array3d(env.screen)
    obs = np.transpose(arr, (1, 0, 2))
    min_dists = [1000] * 3
    
    for col_idx in range(3):
        x = col_centers[col_idx]
        expected_color = note_colors[col_idx]
        for y in hit_window_ys:
            color = obs[y, x]
            dist_color = np.linalg.norm(color - expected_color)
            if dist_color < tolerance:
                dist_to_target = abs(y - 320)
                if dist_to_target < min_dists[col_idx]:
                    min_dists[col_idx] = dist_to_target

    action = [0, 0, 0]
    if min_dists[1] <= 25:
        action[1] = 1
        
    left_dist, right_dist = min_dists[0], min_dists[2]
    if left_dist <= 25 and right_dist <= 25:
        action[0] = 3 if left_dist <= right_dist else 4
    elif left_dist <= 25:
        action[0] = 3
    elif right_dist <= 25:
        action[0] = 4
        
    return action