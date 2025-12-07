def policy(env):
    # Strategy: Avoid obstacles by moving horizontally away from the closest threat, otherwise move upward.
    # Use dash for urgent dodges and long jumps when safe to maximize upward progress and reward.
    player_x, player_y = env.player_pos
    player_size = env.player_size
    danger_obstacle = None
    min_dist = float('inf')
    
    for obs in env.obstacles:
        rect = obs['rect']
        if rect.bottom > player_y - 100 and rect.top < player_y + player_size/2:
            if rect.right > player_x - player_size/2 and rect.left < player_x + player_size/2:
                dist = abs(rect.bottom - (player_y - player_size/2))
                if dist < min_dist:
                    min_dist = dist
                    danger_obstacle = obs
                    
    movement = 1  # Default: move up
    space_held = 0
    shift_held = 0
    
    if danger_obstacle:
        if danger_obstacle['rect'].centerx < player_x and player_x < env.WIDTH - player_size/2:
            movement = 4  # Move right
            if min_dist < 50:
                shift_held = 1
        elif danger_obstacle['rect'].centerx >= player_x and player_x > player_size/2:
            movement = 3  # Move left
            if min_dist < 50:
                shift_held = 1
    elif env.jump_cooldown == 0:
        space_held = 1  # Long jump if safe
        
    return [movement, space_held, shift_held]