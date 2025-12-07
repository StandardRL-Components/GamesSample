def policy(env):
    """
    Strategy: Track the ball's predicted y-position when it reaches the paddle's x-coordinate,
    accounting for wall bounces. Move the paddle to align with this position, using a threshold
    to prevent oscillation. Prioritize hitting the ball (immediate reward) over edge bonuses.
    """
    ball_x, ball_y = env.pixel_pos
    vx, vy = env.pixel_vel
    paddle_y = env.paddle_y
    H = env.HEIGHT

    if vx > 0:  # Ball moving toward paddle
        time_to_paddle = (env.PADDLE_X - ball_x) / vx
        predicted_y = ball_y + vy * time_to_paddle
    elif vx < 0:  # Ball moving away, predict after left wall bounce
        time_to_left = (0 - ball_x) / vx
        time_after_bounce = env.PADDLE_X / (-vx)
        predicted_y = ball_y + vy * (time_to_left + time_after_bounce)
    else:
        predicted_y = ball_y

    # Account for top/bottom wall bounces via reflection
    period = 2 * H
    normalized_y = predicted_y % period
    if normalized_y > H:
        normalized_y = period - normalized_y
    target_y = normalized_y

    current_center = paddle_y + env.PADDLE_HEIGHT / 2
    diff = target_y - current_center
    if diff > 10:
        movement = 2  # Down
    elif diff < -10:
        movement = 1  # Up
    else:
        movement = 0  # No movement

    return [movement, 0, 0]