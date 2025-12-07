def policy(env):
    # Strategy: Track the lowest block above the paddle and align horizontally to intercept it.
    # Prioritize blocks closest to the bottom to prevent game over, then target higher-value blocks when safe.
    # Use minimal movement to avoid oscillation, with a1 and a2 always 0 since they're unused in this environment.
    
    if env.game_over:
        return [0, 0, 0]
    
    paddle_center = env.paddle.x + env.paddle.width / 2
    blocks_above = [b for b in env.blocks if b['rect'].bottom < env.paddle.top]
    
    if not blocks_above:
        # No immediate threats, center the paddle for new blocks
        screen_center = env.WIDTH / 2
        if abs(paddle_center - screen_center) < 5:
            return [0, 0, 0]
        return [3 if paddle_center > screen_center else 4, 0, 0]
    
    # Find the most urgent block (lowest position)
    target_block = max(blocks_above, key=lambda b: b['rect'].y)
    target_x = target_block['rect'].x + target_block['rect'].width / 2
    
    if abs(paddle_center - target_x) < 5:
        return [0, 0, 0]
    return [3 if paddle_center > target_x else 4, 0, 0]