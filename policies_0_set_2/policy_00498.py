def policy(env):
    # Strategy: Center the falling block on the top block to maximize stability and avoid collapse.
    # If already centered, drop the block to build height and progress toward the target.
    if env.game_over or env.game_won:
        return [0, 0, 0]
    
    top_block = env.placed_blocks[-1]
    falling_block = env.falling_block
    
    top_center_x = top_block['center'][0]
    top_center_z = top_block['center'][2]
    falling_center_x = falling_block['pos'][0] + falling_block['size'][0] / 2
    falling_center_z = falling_block['pos'][2] + falling_block['size'][2] / 2
    
    dx = top_center_x - falling_center_x
    dz = top_center_z - falling_center_z
    
    if abs(dx) < 0.1 and abs(dz) < 0.1:
        return [0, 1, 0]
    
    if abs(dx) > abs(dz):
        return [4 if dx > 0 else 3, 0, 0]
    else:
        return [2 if dz > 0 else 1, 0, 0]