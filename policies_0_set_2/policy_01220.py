def policy(env):
    # Strategy: Prioritize survival by maintaining distance from zombies while shooting when safe. 
    # Collect ammo when low, and kite zombies to avoid damage. Shoot only when aimed at zombies to conserve ammo.
    player = env.player
    zombies = env.zombies
    ammo_boxes = env.ammo_boxes
    
    # Find closest zombie and distance
    closest_zombie = None
    min_zombie_dist = float('inf')
    for z in zombies:
        dist = z.pos.distance_to(player.pos)
        if dist < min_zombie_dist:
            min_zombie_dist = dist
            closest_zombie = z
            
    # Find closest ammo box and distance
    closest_ammo = None
    min_ammo_dist = float('inf')
    for a in ammo_boxes:
        dist = a.pos.distance_to(player.pos)
        if dist < min_ammo_dist:
            min_ammo_dist = dist
            closest_ammo = a

    # Movement logic: escape zombies if too close, else seek ammo if low
    a0 = 0
    if closest_zombie and min_zombie_dist < 50:
        dx = player.pos.x - closest_zombie.pos.x
        dy = player.pos.y - closest_zombie.pos.y
        if abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 2 if dy > 0 else 1
    elif player.ammo < 10 and closest_ammo:
        dx = closest_ammo.pos.x - player.pos.x
        dy = closest_ammo.pos.y - player.pos.y
        if abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 2 if dy > 0 else 1

    # Shooting logic: only shoot if aimed at zombie and not on cooldown
    a1 = 0
    if player.ammo > 0 and player.shoot_cooldown == 0 and closest_zombie:
        to_zombie = (closest_zombie.pos - player.pos).normalize()
        if player.aim_vector.dot(to_zombie) > 0.9:  # ~25 degree cone
            a1 = 1

    return [a0, a1, 0]