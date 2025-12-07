def policy(env):
    """
    Jump on beat when next platform is within range, prioritizing risky landings for maximum reward.
    Use high jumps for upward gaps and low jumps for downward gaps, with horizontal adjustments only when necessary.
    """
    # Only jump when grounded and cooldown has passed
    if not env.is_grounded or (env.steps - env.last_action_time) <= 5:
        return [0, 0, 0]
    
    # Find current platform (player's feet at player_pos.y + PLAYER_SIZE)
    current_platform = None
    player_feet_y = env.player_pos.y + env.PLAYER_SIZE
    for p in env.platforms:
        if p.rect.collidepoint(env.player_pos.x, player_feet_y):
            current_platform = p
            break
    
    if current_platform is None:
        return [0, 0, 0]
    
    # Find next platform (first platform ahead of current)
    next_platform = None
    for p in env.platforms:
        if p.rect.left > current_platform.rect.right:
            next_platform = p
            break
    
    if next_platform is None:
        return [0, 0, 0]
    
    # Calculate gap and vertical difference
    gap = next_platform.rect.left - current_platform.rect.right
    vertical_diff = current_platform.rect.y - next_platform.rect.y
    
    # Jump when gap is reasonable and on beat (beat_progress near 0)
    if 50 < gap < 200 and env.beat_progress < 5:
        # Choose jump type based on vertical difference
        if vertical_diff < -10:  # Next platform is higher
            return [0, 1, 0]  # High jump
        elif vertical_diff > 10:  # Next platform is lower
            return [0, 1, 1]  # Low jump
        else:
            return [0, 1, 0]  # Default high jump
    
    return [0, 0, 0]