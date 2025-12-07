def policy(env):
    # Strategy: Always move right to reach exit quickly. Jump when near platform edge to cross pits.
    # On last platform, jump under exit portal. Avoid unnecessary actions (a1/a2 unused in env).
    if env.on_ground:
        current_platform = None
        for plat in env.platforms:
            if plat.left <= env.player_pos.x <= plat.right:
                current_platform = plat
                break
        if current_platform is None:
            return [4, 0, 0]
        last_platform = env.platforms[-1]
        if current_platform == last_platform:
            if env.exit_portal.left <= env.player_pos.x <= env.exit_portal.right:
                return [1, 0, 0]
            elif env.player_pos.x < env.exit_portal.left:
                return [4, 0, 0]
            else:
                return [3, 0, 0]
        else:
            if current_platform.right - env.player_pos.x < 50:
                return [1, 0, 0]
            else:
                return [4, 0, 0]
    else:
        return [4, 0, 0]