def policy(env):
    # Strategy: Track ball's x position and move paddle to intercept, prioritizing edge hits for +40 reward and directing ball toward blocks. When ball is moving up, center paddle to prepare.
    vy = float(env.ball_vel[1])
    ball_y = float(env.ball_pos[1])
    paddle_top = env.paddle_rect.top
    current_center = env.paddle_rect.centerx
    
    if vy > 0 and ball_y < paddle_top:
        left_bound = env.WALL_THICKNESS + env.BALL_RADIUS
        right_bound = env.WIDTH - env.WALL_THICKNESS - env.BALL_RADIUS
        time = (paddle_top - ball_y) / vy
        x = float(env.ball_pos[0])
        vx = float(env.ball_vel[0])
        
        if vx > 0:
            t_wall = (right_bound - x) / vx
            if t_wall < time:
                future_x = right_bound - vx * (time - t_wall)
            else:
                future_x = x + vx * time
        elif vx < 0:
            t_wall = (left_bound - x) / vx
            if t_wall < time:
                future_x = left_bound - vx * (time - t_wall)
            else:
                future_x = x + vx * time
        else:
            future_x = x
            
        left_blocks = sum(1 for b in env.blocks if b['rect'].centerx < env.WIDTH/2)
        right_blocks = len(env.blocks) - left_blocks
        
        if right_blocks > left_blocks:
            target_center = future_x - 40
        elif left_blocks > right_blocks:
            target_center = future_x + 40
        else:
            target_center = future_x - 40
    else:
        target_center = env.WIDTH / 2
        
    left_bound_center = env.WALL_THICKNESS + env.paddle_rect.width/2
    right_bound_center = env.WIDTH - env.WALL_THICKNESS - env.paddle_rect.width/2
    target_center = max(left_bound_center, min(target_center, right_bound_center))
    
    if current_center < target_center - 5:
        a0 = 4
    elif current_center > target_center + 5:
        a0 = 3
    else:
        a0 = 0
        
    return [a0, 0, 0]