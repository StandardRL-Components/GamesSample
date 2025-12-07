def policy(env):
    # Strategy: Track the lowest block to prevent game-over, prioritizing risky edge hits for bonus rewards.
    # Use fast movements for large adjustments and slow for precision. Default to center when no immediate threats.
    if env.game_over:
        return [0, 0, 0]
    
    current_x = env.paddle_rect.x
    critical_height = env.HEIGHT - 100  # Prioritize blocks below this height
    
    # Find the lowest block below critical height, or the overall lowest
    urgent_blocks = [b for b in env.blocks if b['rect'].y >= critical_height]
    target_block = max(urgent_blocks, key=lambda b: b['rect'].y) if urgent_blocks else \
                  max(env.blocks, key=lambda b: b['rect'].y) if env.blocks else None
    
    if target_block:
        block_x = target_block['rect'].x
        block_center = block_x + env.BLOCK_SIZE // 2
        # Adjust target to encourage edge hits if block is near edge
        if block_x <= env.RISKY_EDGE_THRESHOLD:
            target_x = 0  # Left edge for risky hit
        elif block_x >= env.WIDTH - env.RISKY_EDGE_THRESHOLD - env.BLOCK_SIZE:
            target_x = env.WIDTH - env.PADDLE_WIDTH  # Right edge for risky hit
        else:
            target_x = block_center - env.PADDLE_WIDTH // 2  # Center on block
        target_x = max(0, min(target_x, env.WIDTH - env.PADDLE_WIDTH))
    else:
        target_x = env.WIDTH // 2 - env.PADDLE_WIDTH // 2  # Default to center
    
    diff = target_x - current_x
    if diff > 20:
        a0 = 4  # Right fast
    elif diff > 0:
        a0 = 2  # Right slow
    elif diff < -20:
        a0 = 3  # Left fast
    elif diff < 0:
        a0 = 1  # Left slow
    else:
        a0 = 0  # No movement
        
    return [a0, 0, 0]