def policy(env):
    # Strategy: Prioritize rightward movement toward flag and gems. Jump when targets are above or to navigate platforms.
    # Dash only when safe (on solid ground) and facing right to cover distance efficiently. Avoid leftward movement unless
    # gems are very close. Check dash cooldown and grounded state to prevent wasted actions.
    if env.terminated:
        return [0, 0, 0]
    
    player_x, player_y = env.player_pos.x, env.player_pos.y
    targets = []
    
    # Collect all gems and flag as targets
    for gem in env.gems:
        targets.append((gem.centerx, gem.centery))
    if not targets or (targets and player_x > 1400):  # Near flag area
        targets.append((env.flag_pos.x + 5, env.flag_pos.y + 25))  # Flag center
    
    # Find closest target horizontally
    closest_target = None
    min_horizontal_dist = float('inf')
    for tx, ty in targets:
        dist = abs(tx - player_x)
        if dist < min_horizontal_dist:
            min_horizontal_dist = dist
            closest_target = (tx, ty)
    
    if not closest_target:
        return [0, 0, 0]
    
    target_x, target_y = closest_target
    dx = target_x - player_x
    dy = target_y - player_y
    
    # Movement logic
    move_action = 0
    if dx < -10 and abs(dx) < 80:  # Left only if very close
        move_action = 3
    elif dx > 10 or (not env.gems and target_x > player_x):  # Right by default
        move_action = 4
    
    # Jump if target is above or to avoid pits
    jump_action = 1 if (dy < -15 and env.is_grounded) or (env.is_grounded and player_y > 300) else move_action
    
    # Dash only when safe (on ground), facing right, and target is far
    dash_action = 0
    if (env.is_grounded and env.dash_cooldown_timer <= 0 and env.is_dashing <= 0 and
        move_action == 4 and abs(dx) > 120):
        dash_action = 1
    
    return [jump_action, dash_action, 0]