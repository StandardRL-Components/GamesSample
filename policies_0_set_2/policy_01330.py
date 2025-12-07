def policy(env):
    # Strategy: Prioritize reaching the highest platform by jumping toward the next platform's horizontal center.
    # Use power jumps (a2=1) for maximum height, and adjust horizontal movement based on platform alignment.
    # When airborne, continue moving toward target platform; when on ground, jump toward next platform.
    if env.game_over:
        return [0, 0, 0]
    
    on_ground = env.player['on_ground']
    highest_idx = env.highest_platform_index
    next_idx = min(highest_idx + 1, len(env.platforms) - 1)
    next_plat = env.platforms[next_idx]
    dx = next_plat.centerx - env.player['x']
    
    if on_ground:
        if abs(dx) < 15:
            return [1, 0, 1]  # Jump straight up if aligned
        elif dx < 0:
            return [3, 0, 1]  # Jump left if platform is left
        else:
            return [4, 0, 1]  # Jump right if platform is right
    else:
        if dx < -5:
            return [3, 0, 1]  # Air control left
        elif dx > 5:
            return [4, 0, 1]  # Air control right
        else:
            return [0, 0, 1]  # Maintain current trajectory