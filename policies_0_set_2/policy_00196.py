def policy(env):
    # Strategy: Move cursor to blocks not on target, then push towards their target.
    # This greedy approach minimizes distance to targets and maximizes rewards by completing blocks.
    if env.block_animations:
        return [0, 0, 0]  # Wait during animations
    
    # Find first block not on target
    target_block = None
    for block in env.blocks:
        on_target = any(t['id'] == block['id'] and t['pos'] == block['pos'] for t in env.targets)
        if not on_target:
            target_block = block
            break
            
    if not target_block:
        return [0, 0, 0]  # All blocks on target
        
    # Move cursor toward target block
    cx, cy = env.cursor_pos
    bx, by = target_block['pos']
    dx = (bx - cx) % env.GRID_WIDTH
    dx = dx if dx <= env.GRID_WIDTH//2 else dx - env.GRID_WIDTH
    dy = (by - cy) % env.GRID_HEIGHT
    dy = dy if dy <= env.GRID_HEIGHT//2 else dy - env.GRID_HEIGHT
    
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1
        
    # Push if cursor on block and moving toward target
    if cx == bx and cy == by:
        # Find target position
        target_pos = next(t['pos'] for t in env.targets if t['id'] == target_block['id'])
        tx, ty = target_pos
        push_dx = (tx - bx) % env.GRID_WIDTH
        push_dx = push_dx if push_dx <= env.GRID_WIDTH//2 else push_dx - env.GRID_WIDTH
        push_dy = (ty - by) % env.GRID_HEIGHT
        push_dy = push_dy if push_dy <= env.GRID_HEIGHT//2 else push_dy - env.GRID_HEIGHT
        
        if abs(push_dx) > abs(push_dy):
            desired_dir = 4 if push_dx > 0 else 3
        else:
            desired_dir = 2 if push_dy > 0 else 1
            
        if env.last_move_direction == desired_dir and not env.space_was_held:
            return [0, 1, 0]  # Push in desired direction
            
    return [movement, 0, 0]