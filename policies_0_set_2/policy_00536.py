def policy(env):
    # Strategy: Prioritize maintaining optimal track height (y=100) to maximize speed and progress.
    # Extend track with flat segments when at target height, use slopes for height adjustments.
    # Draw only when sled is within 200px of track end to minimize line-drawing penalty.
    if env.last_line_end[0] >= env.finish_line_x:
        return [0, 0, 0]
    if env.sled_pos[0] <= env.last_line_end[0] - 200:
        return [0, 0, 0]
    current_y = env.last_line_end[1]
    if current_y > 120:
        return [1, 0, 0]
    elif current_y < 50:
        return [0, 0, 1]
    elif current_y < 80:
        return [2, 0, 0]
    else:
        return [0, 1, 0]