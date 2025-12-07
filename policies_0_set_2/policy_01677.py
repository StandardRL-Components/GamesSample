def policy(env):
    # Strategy: Prioritize survival by avoiding zombies while shooting when safe. Collect ammo when low.
    # Movement balances repulsion from zombies and attraction to ammo when needed, using inverse square forces.
    # Shooting activates when zombies are aligned with aim direction within a cone and distance threshold.
    px, py = env.player_rect.center
    repulsion = [0.0, 0.0]
    for zombie in env.zombies:
        zx, zy = zombie['rect'].center
        dx, dy = px - zx, py - zy
        dist = max(1.0, (dx*dx + dy*dy)**0.5)
        repulsion[0] += dx / (dist * dist)
        repulsion[1] += dy / (dist * dist)
    
    attraction = [0.0, 0.0]
    if env.player_ammo < 20:
        for pack in env.ammo_packs:
            ax, ay = pack.center
            dx, dy = ax - px, ay - py
            dist = max(1.0, (dx*dx + dy*dy)**0.5)
            attraction[0] += dx / (dist * dist)
            attraction[1] += dy / (dist * dist)
    
    desired_dx = repulsion[0] + attraction[0]
    desired_dy = repulsion[1] + attraction[1]
    norm = max(1e-5, (desired_dx*desired_dx + desired_dy*desired_dy)**0.5)
    desired_dx /= norm
    desired_dy /= norm
    
    movement_candidates = [(0,0), (0,-1), (0,1), (-1,0), (1,0)]
    best_score = -float('inf')
    best_action = 0
    for idx, (dx, dy) in enumerate(movement_candidates):
        score = desired_dx * dx + desired_dy * dy
        if score > best_score:
            best_score = score
            best_action = idx
    
    shoot = 0
    if env.player_ammo > 0 and env.shoot_cooldown == 0:
        fx, fy = env.last_move_direction
        for zombie in env.zombies:
            zx, zy = zombie['rect'].center
            dx, dy = zx - px, zy - py
            dist_sq = dx*dx + dy*dy
            if dist_sq > 10000:  # 100px threshold
                continue
            dot = (fx*dx + fy*dy) / max(1e-5, dist_sq**0.5)
            if dot > 0.9:  # ~25° cone
                shoot = 1
                break
    
    return [best_action, shoot, 0]