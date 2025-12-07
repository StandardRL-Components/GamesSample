def policy(env):
    # Strategy: Alternate between left (col 4) and right (col 5) placement to build a stable
    # centered tower. Use soft drop when aligned to accelerate placement. This maximizes
    # stability by maintaining balanced weight distribution and efficiently builds height.
    if env.falling_tile is None:
        return [0, 0, 0]
    
    target_col = 4 if len(env.placed_tiles) % 2 == 0 else 5
    current_col = env.falling_tile['gx']
    
    if current_col < target_col:
        return [4, 0, 0]
    elif current_col > target_col:
        return [3, 0, 0]
    else:
        return [2, 0, 0]