def policy(env):
    # Strategy: Maximize horizontal progress while avoiding falls. Jump when gaps are detected or coins are above.
    # Prioritize rightward movement for progress reward, jump when platform edges or collectible coins are detected.
    action = [4, 0, 0]  # Default: move right with no secondary actions
    
    # Check if player is on ground and should jump
    if env.on_ground:
        # Detect gaps by checking if platform extends beyond player's right edge
        player_right = env.player_pos[0] + env.player_size[0]
        player_bottom = env.player_pos[1] + env.player_size[1]
        gap_threshold = 30  # pixels to look ahead for platform
        
        # Check for platform continuity
        platform_ahead = any(
            plat.left <= player_right + gap_threshold and 
            plat.right > player_right and 
            plat.top <= player_bottom + 5 and 
            plat.bottom >= player_bottom
            for plat in env.platforms
        )
        
        # Check for coins above to collect
        coins_above = any(
            coin['rect'].bottom < env.player_pos[1] and 
            abs(coin['rect'].centerx - env.player_pos[0]) < 50
            for coin in env.coins
        )
        
        if not platform_ahead or coins_above:
            action[0] = 1  # Jump if gap detected or coin above
    
    return action