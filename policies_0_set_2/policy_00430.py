def policy(env):
    """
    Navigates towards the flag while collecting nearby coins and avoiding pits. 
    Prioritizes immediate rewards (coins) while making progress toward the flag. 
    Jumps when coins are above or when pits are detected ahead to avoid falling.
    """
    if env.game_over:
        return [0, 0, 0]
    
    player_x = env.player_rect.centerx
    player_y = env.player_rect.centery
    
    # Target the flag by default
    target_x = env.flag_rect.centerx
    target_y = env.flag_rect.centery
    jump = False
    
    # Find nearest coin within reasonable range
    coin_target = None
    min_dist = float('inf')
    for coin in env.coins:
        dx = coin.centerx - player_x
        dy = coin.centery - player_y
        if abs(dx) < 100 and abs(dy) < 80:  # Consider coins within range
            dist = abs(dx) + abs(dy)
            if dist < min_dist:
                min_dist = dist
                coin_target = coin
    
    # Prioritize coin collection when available
    if coin_target is not None:
        target_x = coin_target.centerx
        target_y = coin_target.centery
        # Jump if coin is significantly above player
        if target_y < player_y - 15 and env.on_ground:
            jump = True
    
    # Check for impending pits
    for pit in env.pits:
        pit_left, pit_right = pit.left, pit.right
        # Jump if pit is immediately ahead and we're on ground
        if (pit_left - 10 <= player_x <= pit_right and 
            env.on_ground and player_x < env.flag_rect.centerx):
            jump = True
            break
    
    # Move toward target
    if target_x > player_x + 5:
        mov = 4  # Right
    elif target_x < player_x - 5:
        mov = 3  # Left
    else:
        mov = 0  # No movement
    
    return [mov, 1 if jump else 0, 0]