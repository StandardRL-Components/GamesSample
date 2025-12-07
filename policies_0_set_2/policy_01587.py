def policy(env):
    # Strategy: Maximize forward progress by building track aligned with sled's vertical velocity.
    # Use long pieces for coverage, avoid boosters to prevent instability. Prioritize downward
    # slopes when falling to catch sled, upward when rising to extend jumps, else flat for speed.
    if env.game_over:
        return [0, 0, 0]
    vx, vy = env.sled_vel[0], env.sled_vel[1]
    speed_sq = vx * vx + vy * vy
    if speed_sq < 4:
        a0 = 4
    elif vy > 2:
        a0 = 4
    elif vy < -2:
        a0 = 3
    else:
        a0 = 0
    return [a0, 1, 0]