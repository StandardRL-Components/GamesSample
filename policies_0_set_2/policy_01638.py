def policy(env):
    # Strategy: Align current shape with target opening by minimizing x and rotation error.
    # Use fast fall only when aligned to save time. Prioritize rotation then horizontal adjustment.
    if env.game_over or env.current_shape is None:
        return [0, 0, 0]
    
    target = env.level_openings[env.target_opening_index]
    cx, cy = env.current_shape['x'], env.current_shape['y']
    rot = env.current_shape['rot'] % 3.1416  # Normalize to [0, pi)
    
    # Check rotation alignment (target is 0 or pi)
    rot_error = min(rot, 3.1416 - rot)
    if rot_error > 0.1:
        if rot <= 1.57:  # Rotate left if in lower half
            a0 = 2
        else:            # Rotate right if in upper half
            a0 = 1
    else:
        # Adjust horizontal position
        if abs(cx - target['x']) < 2:
            a0 = 0
        elif cx < target['x']:
            a0 = 4
        else:
            a0 = 3
    
    # Use fast fall if aligned (within thresholds)
    a1 = 1 if (rot_error < 0.2 and abs(cx - target['x']) < 15) else 0
    a2 = 0
    
    return [a0, a1, a2]