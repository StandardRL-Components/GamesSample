def policy(env):
    """
    Prioritizes collecting coins for immediate reward, then moves to the flag to complete the level.
    Uses internal state to navigate efficiently: moves horizontally toward nearest uncollected coin (or flag),
    jumps when target is above player and on ground, and avoids unnecessary jumps when falling or on platforms.
    """
    # Get current player position and game state
    player_x, player_y = env.player_pos
    on_ground = env.on_ground
    
    # Find nearest uncollected coin or fall back to flag
    target_x, target_y = env.flag_pos
    min_dist = float('inf')
    for coin in env.coins:
        if not coin['collected']:
            coin_x, coin_y = coin['rect'].center
            dist = abs(coin_x - player_x)
            if dist < min_dist:
                min_dist = dist
                target_x, target_y = coin_x, coin_y
    
    # Determine horizontal movement
    dx = target_x - player_x
    if dx > 10:  # Move right if target is sufficiently to the right
        move_action = 4
    elif dx < -10:  # Move left if target is sufficiently to the left
        move_action = 3
    else:
        move_action = 0  # Stop when close horizontally
    
    # Jump if target is above and on ground, or to overcome small obstacles
    jump_action = 1 if (on_ground and target_y < player_y - 20) else 0
    
    # Secondary action unused in this environment
    return [move_action, jump_action, 0]