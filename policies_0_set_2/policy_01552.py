def policy(env):
    # Strategy: Focus on building a continuous track slightly below and ahead of the sled to maintain momentum and reach the finish.
    # Prioritize drawing when sled is airborne or slowing down, and erase only if track obstructs progress or causes falls.
    if env.game_over:
        return [0, 0, 0]
    
    sled_x, sled_y = env.sled_pos
    cursor_x, cursor_y = env.cursor_pos
    
    # Calculate target position: ahead and below sled to extend track
    target_x = sled_x + 50
    target_y = sled_y + 20
    target_x = max(0, min(env.WIDTH, target_x))
    target_y = max(0, min(env.HEIGHT, target_y))
    
    # Move cursor toward target
    dx = target_x - cursor_x
    dy = target_y - cursor_y
    action0 = 0
    if abs(dx) > 5:
        action0 = 4 if dx > 0 else 3
    elif abs(dy) > 5:
        action0 = 2 if dy > 0 else 1
    
    # Determine if drawing or erasing is needed
    action1 = 0
    action2 = 0
    if not env.on_surface or env.sled_vel[0] < 1.0:
        action1 = 1  # Draw track if sled is falling or slow
    # Erase only if cursor is near sled and there's existing track that may be causing issues
    if (abs(cursor_x - sled_x) < 30 and abs(cursor_y - sled_y) < 30 and
        env.track_segments and not env.on_surface):
        action2 = 1
    
    return [action0, action1, action2]