def policy(env):
    """
    Strategy: Maximize stability and height by aligning falling block with current center of mass (COM) 
    to maintain balance. Fast drop only when aligned to minimize misplacement risk. COM calculations 
    ensure tower stability while platform-centered targeting minimizes offset penalty.
    """
    if env.game_over or env.falling_block is None:
        return [0, 0, 0]
    
    n = len(env.stacked_blocks)
    current_com = env.center_of_mass_x
    platform = env.platform
    block_width = env.BLOCK_WIDTH
    
    # Calculate safe placement range to keep COM within platform
    L = platform.left * (n + 1) - n * current_com
    R = platform.right * (n + 1) - n * current_com
    desired_x = platform.centerx
    target_x = max(L, min(R, desired_x))
    target_x = max(platform.left + block_width / 2, min(platform.right - block_width / 2, target_x))
    
    current_x = env.falling_block['rect'].centerx
    threshold = 1
    
    if current_x < target_x - threshold:
        move_action = 4  # right
    elif current_x > target_x + threshold:
        move_action = 3  # left
    else:
        move_action = 0
    
    fast_drop = 1 if abs(current_x - target_x) <= threshold else 0
    return [move_action, fast_drop, 0]