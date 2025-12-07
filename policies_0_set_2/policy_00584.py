def policy(env):
    # Strategy: Track ball position and align paddle to intercept. Launch ball immediately when on paddle.
    # Prioritizes preventing life loss by catching the ball, then breaking blocks for score.
    obs = env.render()
    paddle_center = 320
    count = 0
    total_x = 0
    for y in range(375, 391):
        for x in range(640):
            pixel = obs[y, x]
            if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255:
                total_x += x
                count += 1
    if count > 0:
        paddle_center = total_x / count

    ball_x = None
    ball_y = None
    ball_pixels = []
    for y in range(0, 375):
        for x in range(0, 640, 2):
            pixel = obs[y, x]
            if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 0:
                ball_pixels.append((x, y))
    if not ball_pixels:
        for y in range(375, 400):
            for x in range(0, 640, 2):
                pixel = obs[y, x]
                if pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 0:
                    ball_pixels.append((x, y))
    if ball_pixels:
        total_x = 0
        total_y = 0
        for (x, y) in ball_pixels:
            total_x += x
            total_y += y
        ball_x = total_x / len(ball_pixels)
        ball_y = total_y / len(ball_pixels)

    on_paddle = False
    if ball_x is not None and ball_y is not None:
        if abs(ball_y - 367) < 5 and abs(ball_x - paddle_center) < 50:
            on_paddle = True

    if on_paddle:
        return [0, 1, 0]
    elif ball_x is None:
        return [0, 0, 0]
    else:
        if ball_x < paddle_center - 10:
            return [3, 0, 0]
        elif ball_x > paddle_center + 10:
            return [4, 0, 0]
        else:
            return [0, 0, 0]