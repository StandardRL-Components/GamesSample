def policy(env):
    """
    Strategy: Jump only when on ground and after a brief delay on the first platform to allow potential horizontal movement (if any).
    For non-initial platforms, jump immediately to avoid penalty and maximize progression.
    This addresses the previous policy's failure by allowing time for platform alignment before jumping.
    """
    if env.on_ground:
        if env.last_platform_landed == env.platforms[0]:
            if env.steps > 10:
                return [0, 1, 0]
            else:
                return [0, 0, 0]
        else:
            return [0, 1, 0]
    else:
        return [0, 0, 0]