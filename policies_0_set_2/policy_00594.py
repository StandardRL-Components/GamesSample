def policy(env):
    # Strategy: Prioritize moving right to maximize progress and coin collection. Jump when coins are directly above or when approaching platform edges to avoid gaps. Always use primary and secondary actions conservatively (set to 0) as they are unused in this environment.
    action = [4, 0, 0]  # Default: move right
    
    # Jump if on ground and coin is directly above within range
    if env.on_ground and env.can_jump:
        player_mid_x = env.player_pos.x + 12
        for coin in env.coins:
            if not coin["active"]:
                continue
            coin_rect = coin["rect"]
            # Check if coin is horizontally aligned and above player
            if (abs(coin_rect.centerx - player_mid_x) < 20 and 
                coin_rect.bottom < env.player_pos.y):
                action[0] = 1  # Jump
                break
    
    return action