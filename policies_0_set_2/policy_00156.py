def policy(env):
    # Strategy: Always jump when on ground to ascend quickly. Move horizontally to align with next platform.
    # Use fast fall when above a platform to land faster. This maximizes upward progress and reward.
    action = [0, 0, 0]  # [movement, space, shift]
    
    # Get next platform target if available
    next_plat_idx = env.highest_platform_idx + 1
    if next_plat_idx < len(env.platforms):
        target_x = env.platforms[next_plat_idx].centerx
    else:
        target_x = env.WIDTH / 2  # Default to center if no platform
    
    # Horizontal movement towards target
    if env.player_pos.x < target_x - 10:
        action[0] = 4  # Right
    elif env.player_pos.x > target_x + 10:
        action[0] = 3  # Left
    
    # Jump when on ground
    if env.on_ground:
        action[1] = 1
    
    # Fast fall when above target platform and falling
    if (not env.on_ground and 
        next_plat_idx < len(env.platforms) and
        env.platforms[next_plat_idx].collidepoint(env.player_pos.x, env.platforms[next_plat_idx].y) and
        env.player_vel.y > 0):
        action[2] = 1
        
    return action