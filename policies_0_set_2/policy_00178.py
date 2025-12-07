def policy(env):
    # Strategy: Always move right to progress, jump over pits by detecting gaps in ground platforms,
    # and jump for coins above. Uses player position and environment state to make decisions.
    # This achieves high reward by maximizing forward progress and coin collection while avoiding falls.
    action = [4, 0, 0]  # Default: move right, no jump
    
    # Check for pits ahead within jump range
    player_x = env.player_pos.x
    player_y = env.player_pos.y
    ground_y = env.GROUND_Y
    
    # Look for pits starting within 50-150 pixels ahead
    for pit in env.pits:
        if 50 <= pit.left - player_x <= 150:
            # Jump if grounded and pit is detected ahead
            if env.is_grounded or env.coyote_time > 0:
                action = [4, 1, 0]
            break
    
    # Check for coins above and within collection range
    for coin in env.coins:
        if (abs(coin.centerx - player_x) < 50 and 
            player_y - coin.centery > 30 and
            (env.is_grounded or env.coyote_time > 0)):
            action = [4, 1, 0]
            break
    
    return action