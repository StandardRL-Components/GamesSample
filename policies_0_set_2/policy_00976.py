def policy(env):
    # Strategy: Prioritize catching fruits closest to the bottom to prevent life loss.
    # Track the most urgent fruit (lowest vertical position) and move to intercept it.
    # Use a dynamic tolerance based on fruit fall speed to minimize oscillation.
    if env.fruits:
        # Find the most urgent fruit (lowest on screen)
        target_fruit = max(env.fruits, key=lambda f: f['rect'].y)
        target_x = target_fruit['rect'].centerx
        catcher_center = env.catcher_x + env.catcher_width / 2
        # Dynamic tolerance based on catcher speed and fruit fall rate
        tolerance = max(15, env.catcher_speed - env.fruit_fall_speed)
        
        if abs(target_x - catcher_center) > tolerance:
            if target_x < catcher_center:
                return [3, 0, 0]  # Move left
            else:
                return [4, 0, 0]  # Move right
    return [0, 0, 0]  # No movement needed