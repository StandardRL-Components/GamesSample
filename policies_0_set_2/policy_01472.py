def policy(env):
    # Strategy: Track the ball's x-position and move the paddle to align with it. This maximizes
    # the chance of hitting the ball, breaking bricks (reward), and preventing ball loss (penalty).
    ball_x = env.ball_pos[0]
    paddle_center = env.paddle.centerx
    if paddle_center < ball_x - 5:
        movement = 4  # Move right
    elif paddle_center > ball_x + 5:
        movement = 3  # Move left
    else:
        movement = 0  # Hold position
    return [movement, 0, 0]