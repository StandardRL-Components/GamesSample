def policy(env):
    # Strategy: Predict ball's landing x and align paddle center, using tolerance to minimize oscillation.
    # Prioritizes hitting the ball (avoiding life loss) and opportunistically achieves risky bounces by natural edge alignment.
    predicted_x = env._predict_ball_landing_x()
    if predicted_x is None:
        return [0, 0, 0]
    paddle_center = env.paddle_pos.centerx
    if paddle_center < predicted_x - 5:
        return [4, 0, 0]
    elif paddle_center > predicted_x + 5:
        return [3, 0, 0]
    else:
        return [0, 0, 0]