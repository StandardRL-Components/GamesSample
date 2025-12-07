def policy(env):
    # Strategy: Move right to reach goal flag quickly, jump over obstacles and gaps when detected,
    # and use boost mid-air to extend jumps when needed. Prioritize forward progress to maximize reward.
    img = env._get_observation()
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    player_color = (50, 150, 255)
    platform_color = (100, 200, 120)
    obstacle_color = (255, 80, 80)
    
    # Find player position (approximate center region)
    player_x, player_y = center_x, center_y
    for y in range(center_y-15, center_y+15):
        for x in range(center_x-15, center_x+15):
            if all(abs(int(img[y,x,i]) - player_color[i]) < 20 for i in range(3)):
                player_x, player_y = x, y
                break
    
    # Check for ground below player
    on_ground = False
    if player_y + 25 < h:
        for dx in range(-10, 10):
            x_check = player_x + dx
            if 0 <= x_check < w:
                if all(abs(int(img[player_y+25, x_check, i]) - platform_color[i]) < 20 for i in range(3)):
                    on_ground = True
                    break
    
    # Check for obstacle ahead
    obstacle_ahead = False
    for dx in range(10, 30):
        x_check = player_x + dx
        if x_check < w:
            for dy in range(-15, 15):
                y_check = player_y + dy
                if 0 <= y_check < h:
                    if all(abs(int(img[y_check, x_check, i]) - obstacle_color[i]) < 20 for i in range(3)):
                        obstacle_ahead = True
                        break
    
    # Check for gap ahead
    gap_ahead = False
    if player_y + 25 < h:
        for dx in range(15, 35):
            x_check = player_x + dx
            if x_check < w:
                if all(abs(int(img[player_y+25, x_check, i]) - platform_color[i]) > 40 for i in range(3)):
                    gap_ahead = True
                    break
    
    # Action logic
    movement = 4  # Default: move right
    space_held = 0
    secondary = 0
    
    if on_ground and (obstacle_ahead or gap_ahead):
        movement = 1  # Jump if obstacle/gap detected
    elif not on_ground:
        space_held = 1  # Use boost mid-air when available
    
    return [movement, space_held, secondary]