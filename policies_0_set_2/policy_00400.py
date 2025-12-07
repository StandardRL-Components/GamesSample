def policy(env):
    # Strategy: Center the falling block and drop to maximize stability and height gain.
    # Scan the top half of the screen for the falling block (bright colors) to determine its x-position.
    # Move horizontally to center it (x=320) and drop when aligned to build a stable tower.
    obs = env._get_observation()
    height, width, _ = obs.shape
    bright_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 165, 0), (0, 255, 0)]
    found_x = None
    for y in range(0, 200):
        for x in range(0, width):
            pixel = tuple(obs[y, x])
            if pixel in bright_colors:
                found_x = x
                break
        if found_x is not None:
            break
    if found_x is None:
        return [0, 0, 0]
    if found_x < 310:
        return [4, 0, 0]
    elif found_x > 330:
        return [3, 0, 0]
    else:
        return [0, 1, 0]