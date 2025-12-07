def policy(env):
    # Strategy: Always move right to reach flag quickly. Jump when approaching platform edges or tokens above.
    # Dash when tokens available and facing a large gap to maximize horizontal movement efficiency.
    # Prioritize token collection for dash fuel and reward, but avoid unnecessary detours.
    action = [4, 0, 0]  # Default: move right, no jump/dash
    
    # Check if player is on ground
    on_ground = env.on_ground
    
    # Find current platform
    current_platform = None
    player_rect = env.player_rect
    for plat in env.platforms:
        if player_rect.colliderect(plat) and player_rect.bottom == plat.top:
            current_platform = plat
            break
    
    # Jump if at platform edge or token above
    if on_ground:
        # Check for platform edge
        if current_platform and (current_platform.right - player_rect.right < 15):
            action[0] = 1  # Jump
        
        # Check for tokens above
        for token in env.risk_tokens:
            if not token['collected']:
                token_pos = token['pos']
                if (abs(token_pos.x - player_rect.centerx) < 30 and 
                    player_rect.top - token_pos.y < 50):
                    action[0] = 1  # Jump
                    break
    
    # Dash if tokens available and facing large gap
    if env.risk_tokens_collected > 0:
        # Find next platform
        next_platform = None
        for plat in env.platforms:
            if plat.left > player_rect.right:
                if next_platform is None or plat.left < next_platform.left:
                    next_platform = plat
        
        # Dash if gap is large
        if next_platform and (next_platform.left - player_rect.right > 80):
            action[2] = 1  # Dash
    
    return action