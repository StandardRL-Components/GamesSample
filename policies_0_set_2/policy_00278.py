def policy(env):
    # Strategy: Maximize forward progress and checkpoint collection by building tracks
    # that guide the sled toward targets while preventing falls. Prioritize drawing
    # toward the next checkpoint or finish, using long lines for efficiency. When
    # the sled is near the bottom, emergency draw downward to catch it.
    if env.game_over:
        return [0, 0, 0]
    
    sled_x, sled_y = env.sled_pos
    if sled_y > env.SCREEN_HEIGHT - 50:
        target = (sled_x, env.SCREEN_HEIGHT)
    else:
        if env.unclaimed_checkpoints:
            target = env.unclaimed_checkpoints[0]
        else:
            target = env.finish_pos
            
    drawing_x, drawing_y = env.last_draw_point
    target_x, target_y = target
    
    if target_x > drawing_x:
        dx = target_x - drawing_x
        dy = abs(target_y - drawing_y)
        if target_y > drawing_y:
            if dy > dx:
                a0 = 2
            else:
                a0 = 4
        else:
            if dy > dx:
                a0 = 1
            else:
                a0 = 4
    else:
        if target_y > drawing_y:
            a0 = 2
        else:
            a0 = 1
            
    return [a0, 1, 1]