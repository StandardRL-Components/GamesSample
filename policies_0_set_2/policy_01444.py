def policy(env):
    """
    Strategy: Maximizes reward by dashing to the next right shadow only when safely positioned at the center of the current shadow, ensuring minimal exposure to guards. This prioritizes progression toward the exit while maintaining stealth.
    """
    if env.game_over:
        return [0, 0, 0]
    
    in_shadow = env._is_player_in_shadow()
    dx = env.player_pos[0] - env.player_target_pos[0]
    dy = env.player_pos[1] - env.player_target_pos[1]
    dist_sq = dx * dx + dy * dy
    
    if in_shadow and dist_sq < 1.0:
        player_x = env.player_pos[0]
        found_right_shadow = any(env._get_shadow_center(shadow)[0] > player_x + 20 for shadow in env.shadows)
        if found_right_shadow and not env.space_pressed_last_frame:
            return [0, 1, 0]
    
    return [0, 0, 0]