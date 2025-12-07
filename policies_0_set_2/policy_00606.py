def policy(env):
    # Strategy: During aiming phase, adjust launcher angle to target the lowest available block (prioritizing center blocks for optimal bounces) and launch when aligned. During playing phase, wait for balls to settle to avoid wasting actions.
    if env.game_phase == "playing":
        return [0, 0, 0]
    
    if not env.blocks:
        return [0, 1, 0]
    
    center_x = env.WIDTH // 2
    lowest_y = -1
    candidate_blocks = []
    for block in env.blocks:
        block_y = block['rect'].centery
        if block_y > lowest_y:
            lowest_y = block_y
            candidate_blocks = [block]
        elif block_y == lowest_y:
            candidate_blocks.append(block)
    
    best_block = min(candidate_blocks, key=lambda b: abs(b['rect'].centerx - center_x))
    target_x = best_block['rect'].centerx
    error = target_x - center_x
    
    if abs(error) < 20:
        return [0, 1, 0]
    else:
        return [1 if error < 0 else 2, 0, 0]