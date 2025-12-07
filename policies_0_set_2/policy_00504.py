def policy(env):
    # Strategy: Always move right to reach the flag quickly, jumping when coins are above or when at platform edges to avoid falling.
    # Prioritize coin collection for immediate reward while maintaining progress toward the goal.
    player_rect = env.player['rect']
    on_ground = env.player['on_ground']
    
    # Check for coins above within jump range
    coin_above = False
    for coin in env.coins:
        if (abs(coin.centerx - player_rect.centerx) < 50 and 
            player_rect.top - coin.bottom < 100 and 
            player_rect.top > coin.bottom):
            coin_above = True
            break
    
    # Check if near right edge of current platform
    near_edge = False
    if on_ground:
        for plat in env.platforms:
            if (player_rect.bottom == plat.top and 
                player_rect.right >= plat.left and 
                player_rect.left <= plat.right and
                plat.right - player_rect.right < 10):
                near_edge = True
                break
    
    # Jump if coin above or near platform edge, otherwise move right
    if on_ground and (coin_above or near_edge):
        return [4, 1, 0]  # Move right and jump
    else:
        return [4, 0, 0]  # Move right only