def policy(env):
    # Strategy: Adjust angle to 75° and power to 100 for optimal arc to reach the hoop at (540,150)
    # from start position (100,300). This maximizes scoring probability by ensuring sufficient
    # height and distance while avoiding backboard collisions.
    if env.ball_state != "AIMING":
        return [0, 0, 0]
    
    if env.shot_clock <= 5:
        return [0, 1, 0]
    
    target_angle = 75
    target_power = 100
    
    if env.aim_angle < target_angle:
        return [1, 0, 0]
    elif env.aim_angle > target_angle:
        return [2, 0, 0]
    elif env.aim_power < target_power:
        return [4, 0, 0]
    elif env.aim_power > target_power:
        return [3, 0, 0]
    else:
        return [0, 1, 0]