def policy(env):
    # Strategy: Track ball position and align paddle underneath when ball is descending.
    # Launch immediately when ball is on paddle. Maximizes blocks broken per ball and prevents losing lives.
    obs = env._get_observation()
    paddle_y_range = range(375, 390)
    ball_color = (255, 255, 0)
    paddle_color = (220, 220, 220)
    
    ball_x, ball_y, paddle_x = None, None, None
    for y in range(400):
        for x in range(640):
            r, g, b = obs[y, x]
            if (r, g, b) == ball_color:
                ball_x, ball_y = x, y
            if y in paddle_y_range and (r, g, b) == paddle_color:
                paddle_x = x
    
    if ball_x is None or paddle_x is None:
        return [0, 0, 0]
    
    if ball_y >= 375:  # Ball on or near paddle
        if abs(ball_x - paddle_x) > 5:
            return [4 if ball_x > paddle_x else 3, 0, 0]
        return [0, 1, 0]  # Launch when aligned
    
    if ball_y > 200:  # Ball in lower half - track it
        if ball_x > paddle_x + 5:
            return [4, 0, 0]
        if ball_x < paddle_x - 5:
            return [3, 0, 0]
    return [0, 0, 0]