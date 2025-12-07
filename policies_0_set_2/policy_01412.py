def policy(env):
    # Aim paddle to align with ball's predicted return position when active, or target block center when inactive.
    # Launch ball when aligned to maximize breaks and chain reactions while avoiding misses.
    a0, a1, a2 = 0, 0, 0
    if not env.ball_active:
        if env.blocks:
            total_x = sum(block['rect'].centerx for block in env.blocks)
            total_y = sum(block['rect'].centery for block in env.blocks)
            target_x = total_x / len(env.blocks)
            target_y = total_y / len(env.blocks)
            dx = target_x - env.paddle_pos.x
            dy = env.paddle_pos.y - target_y
            if abs(dy) > 1e-5:
                desired_angle = (dx / dy) * 57.2958  # rad to deg approximation
                desired_angle = max(min(desired_angle, 75), -75)
                if desired_angle > env.paddle_angle + 2:
                    a0 = 2
                elif desired_angle < env.paddle_angle - 2:
                    a0 = 1
                else:
                    a1 = 1
    else:
        if env.ball_vel.y > 0:
            future_x = env.ball_pos.x + env.ball_vel.x * ((env.PADDLE_Y - env.ball_pos.y) / env.ball_vel.y)
            dx = future_x - env.paddle_pos.x
            desired_angle = (dx / 100) * 75
            desired_angle = max(min(desired_angle, 75), -75)
            if desired_angle > env.paddle_angle + 2:
                a0 = 2
            elif desired_angle < env.paddle_angle - 2:
                a0 = 1
    return [a0, a1, a2]