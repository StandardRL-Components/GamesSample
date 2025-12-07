def policy(env):
    # Strategy: Move reticle towards closest target and fire when aligned. Uses function attribute to track fire state for press (not hold) to avoid ammo waste and maximize hit rate.
    if not hasattr(policy, "prev_fire"):
        policy.prev_fire = False  # Initialize previous fire state
        
    if env.game_over:
        return [0, 0, 0]
    
    # Get current reticle position and targets from environment state
    reticle_pos = env.reticle_pos
    targets = env.targets
    
    if not targets:
        return [0, 0, 0]
    
    # Find closest target to reticle
    closest_target = min(targets, key=lambda t: reticle_pos.distance_to(t['pos']))
    target_pos = closest_target['pos']
    
    # Calculate movement direction toward target
    dx = target_pos.x - reticle_pos.x
    dy = target_pos.y - reticle_pos.y
    movement = 0
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1
    
    # Fire when aligned with target and ammo available
    dist = reticle_pos.distance_to(target_pos)
    fire_now = (dist < closest_target['radius'] + 10) and env.ammo > 0
    fire_action = 1 if (fire_now and not policy.prev_fire) else 0
    policy.prev_fire = fire_action == 1
    
    return [movement, fire_action, 0]