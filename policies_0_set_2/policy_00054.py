def policy(env):
    # Strategy: Move towards the lowest block to avoid missing (costs a life) and catch when aligned.
    # Only catch if space wasn't pressed last frame (rising edge). This maximizes catches while minimizing misses.
    movement = 0
    space_held = 0
    
    # Find the lowest block (closest to bottom)
    blocks_above = [b for b in env.blocks if b['rect'].bottom < env.catcher_pos.top]
    if blocks_above:
        target_block = min(blocks_above, key=lambda b: b['rect'].bottom)
        target_x = target_block['rect'].centerx
        catcher_x = env.catcher_pos.centerx
        
        if target_x < catcher_x - 5:
            movement = 3  # Move left
        elif target_x > catcher_x + 5:
            movement = 4  # Move right
    
    # Catch if colliding with any block and space wasn't pressed last frame
    colliding_blocks = [b for b in env.blocks if env.catcher_pos.colliderect(b['rect'])]
    if colliding_blocks and not env.space_pressed_last_frame:
        space_held = 1
    
    return [movement, space_held, 0]