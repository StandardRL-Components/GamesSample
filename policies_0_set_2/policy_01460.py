def policy(env):
    # Strategy: Track the lowest fruit (most urgent) and move horizontally to align basket center with fruit's x position.
    # This maximizes catch rate by prioritizing imminent fruits while avoiding unnecessary movement (reducing penalty).
    if not env.fruits:
        return [0, 0, 0]
    lowest_fruit = min(env.fruits, key=lambda f: f['pos'][1])
    fruit_x, basket_x = lowest_fruit['pos'][0], env.basket_x
    tolerance = env.BASKET_WIDTH / 2 - env.FRUIT_RADIUS
    if abs(fruit_x - basket_x) <= tolerance:
        return [0, 0, 0]
    return [3 if fruit_x < basket_x else 4, 0, 0]