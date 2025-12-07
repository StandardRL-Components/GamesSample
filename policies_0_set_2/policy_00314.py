def policy(env):
    # Jump on the beat to maximize timing rewards and avoid obstacles. Check for immediate collision threats to prioritize survival.
    beat_period = env.beat_period_frames
    phase = env.steps % beat_period
    on_ground = (env.player_y == env.GROUND_Y - env.PLAYER_HEIGHT)
    
    # Check for imminent collision
    danger = False
    for obs in env.obstacles:
        if obs.x <= 130 and obs.x + obs.width >= 70 and on_ground:
            danger = True
            break
    
    if danger:
        a1 = 1 if (on_ground and not env.is_jumping) else 0
    else:
        # Jump on beat with perfect timing window
        if (phase <= 3 or phase >= beat_period - 3) and on_ground and not env.is_jumping:
            a1 = 1
        else:
            a1 = 0
            
    return [0, a1, 0]