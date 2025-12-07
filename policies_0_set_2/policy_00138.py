def policy(env):
    # Strategy: Always move right to reach the goal faster. Jump when grounded and next platform is higher to progress upward.
    # This maximizes reward by efficiently reaching new platforms (+10) and the goal (+100 + time bonus).
    if env.is_grounded:
        current_idx = env.highest_platform_index
        if current_idx + 1 < len(env.platforms):
            next_platform = env.platforms[current_idx + 1]
            current_platform = env.platforms[current_idx]
            if next_platform.top < current_platform.top:
                return [4, 1, 0]  # Jump if next platform is higher
    return [4, 0, 0]  # Always move right, no jump