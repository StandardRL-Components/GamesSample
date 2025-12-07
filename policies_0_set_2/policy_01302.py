def policy(env):
    # Strategy: Prioritize catching fruits in immediate danger (lowest Y) when catcher is ready, else move to align with lowest fruit to prevent misses.
    if env.game_over:
        return [0, 0, 0]
    
    ready_to_catch = env.catcher_cooldown_timer <= 0 and env.catcher_active_timer <= 0
    catcher_low_y = env.SCREEN_HEIGHT - env.CATCHER_HEIGHT - 10
    catcher_high_y = env.SCREEN_HEIGHT
    catcher_left = env.catcher_pos_x - env.CATCHER_WIDTH / 2
    catcher_right = env.catcher_pos_x + env.CATCHER_WIDTH / 2

    if ready_to_catch:
        for fruit in env.fruits:
            if (catcher_low_y <= fruit['y'] <= catcher_high_y and 
                catcher_left <= fruit['x'] <= catcher_right):
                return [0, 1, 0]

    if env.fruits:
        lowest_fruit = max(env.fruits, key=lambda f: f['y'])
        target_x = lowest_fruit['x']
        if env.catcher_pos_x < target_x - 5:
            return [4, 0, 0]
        elif env.catcher_pos_x > target_x + 5:
            return [3, 0, 0]
        else:
            return [0, 0, 0]
    else:
        center = env.SCREEN_WIDTH / 2
        if abs(env.catcher_pos_x - center) > 5:
            if env.catcher_pos_x < center:
                return [4, 0, 0]
            else:
                return [3, 0, 0]
        return [0, 0, 0]