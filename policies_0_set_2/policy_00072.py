def policy(env):
    # Strategy: Prioritize placing long-range (slow) towers to cover wider areas and eliminate enemies early.
    # Move reticle toward the closest enemy to base to place towers in critical paths, maximizing coverage and damage.
    # This approach efficiently defends the base by targeting high-threat areas and leveraging superior tower range.

    # Check if current reticle position is valid for placing a slow tower
    base_pos = env.base_pos
    reticle_pos = env.reticle_pos
    valid_placement = True
    if (reticle_pos - base_pos).length() < env.BASE_SIZE:
        valid_placement = False
    for tower in env.towers:
        if (reticle_pos - tower['pos']).length() < env.TOWER_PLACEMENT_RADIUS * 2:
            valid_placement = False
            break

    # Place slow tower if valid
    if valid_placement:
        return [0, 0, 1]

    # Find closest enemy to base
    closest_enemy = None
    min_dist = float('inf')
    for enemy in env.enemies:
        dist = (enemy['pos'] - base_pos).length()
        if dist < min_dist:
            min_dist = dist
            closest_enemy = enemy

    # Move toward closest enemy or default pattern if no enemies
    if closest_enemy is not None:
        target_pos = closest_enemy['pos']
    else:
        # Default circular pattern around base
        angle = (env.steps % 360) * (3.14159 / 180)
        target_pos = base_pos + pygame.math.Vector2(100 * math.cos(angle), 100 * math.sin(angle))

    # Calculate movement direction
    dx = target_pos.x - reticle_pos.x
    dy = target_pos.y - reticle_pos.y
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1

    return [movement, 0, 0]