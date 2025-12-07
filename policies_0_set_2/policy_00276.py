def policy(env):
    # Strategy: Maximize score by moving towards high-value fruits (especially golden apples) while avoiding bombs.
    # Use a potential field approach: fruits attract the basket proportional to their value and inverse distance,
    # while bombs repel the basket proportional to their penalty and inverse distance. Move in the direction of net force.
    basket_center = env.basket_x + env.basket_width / 2
    force = 0.0
    for fruit in env.fruits:
        if fruit['y'] < env.SCREEN_HEIGHT - env.basket_height - 10:  # Only consider fruits above basket
            value = 5 if fruit['type'] == 'golden_apple' else 1
            dist_y = (env.SCREEN_HEIGHT - env.basket_height - 10) - fruit['y']
            weight = value / (dist_y + 1)
            force += weight * (fruit['x'] - basket_center)
    for bomb in env.bombs:
        if bomb['y'] < env.SCREEN_HEIGHT - env.basket_height - 10:
            dist_y = (env.SCREEN_HEIGHT - env.basket_height - 10) - bomb['y']
            weight = -10 / (dist_y + 1)
            force += weight * (bomb['x'] - basket_center)
    if force > 5:
        return [4, 0, 0]  # Move right
    elif force < -5:
        return [3, 0, 0]  # Move left
    else:
        return [0, 0, 0]  # No movement