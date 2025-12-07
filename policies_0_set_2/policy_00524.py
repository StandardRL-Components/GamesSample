def policy(env):
    # Strategy: Align falling block with top block's center for safe stacking, then fast-drop.
    # Maximizes score by avoiding falls (termination) while enabling future risky placements.
    if env.game_over:
        return [0, 0, 0]
    
    top_rect, _ = env.stacked_blocks[-1]
    falling_rect = env.falling_block['rect']
    
    target_x = top_rect.centerx - falling_rect.width // 2
    target_x = max(0, min(env.WIDTH - falling_rect.width, target_x))
    
    current_x = falling_rect.x
    if current_x < target_x - 8:
        move = 4  # Right
    elif current_x > target_x + 8:
        move = 3  # Left
    else:
        move = 0  # None
    
    return [move, 1, 0]  # Always fast-drop (a1=1), no secondary action (a2=0)