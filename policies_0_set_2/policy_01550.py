def policy(env):
    # Strategy: Track the lowest fruit above the catcher to prioritize immediate catches, moving horizontally to align the basket.
    # Avoid unnecessary movement by using a tolerance margin (catcher_width/4) to prevent oscillation.
    # Secondary actions (a1, a2) are unused in this environment and set to 0.
    catcher_x = env.catcher_pos[0]
    catcher_top = env.catcher_pos[1] - env.catcher_height
    valid_fruits = [f for f in env.fruits if f['pos'][1] - f['size'] < catcher_top]
    if not valid_fruits:
        return [0, 0, 0]
    target_fruit = max(valid_fruits, key=lambda f: f['pos'][1])
    target_x = target_fruit['pos'][0]
    if target_x < catcher_x - env.catcher_width / 4:
        movement = 3
    elif target_x > catcher_x + env.catcher_width / 4:
        movement = 4
    else:
        movement = 0
    return [movement, 0, 0]