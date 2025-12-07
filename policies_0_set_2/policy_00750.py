def policy(env):
    # Strategy: Prioritize mining closest asteroid while maintaining safe distance from enemies.
    # Use thrust to align with target direction, mine when close, and evade nearby enemies.
    # This maximizes mineral collection and minimizes collision risk for optimal reward.
    import math
    import pygame

    player_pos = env.player.pos
    world_width = env.WORLD_WIDTH
    world_height = env.WORLD_HEIGHT
    safe_dist = 150
    mining_radius = env.MINING_RADIUS

    min_enemy_dist = float('inf')
    closest_enemy = None
    for enemy in env.enemies:
        dx = enemy.pos.x - player_pos.x
        dy = enemy.pos.y - player_pos.y
        if dx > world_width / 2:
            dx -= world_width
        elif dx < -world_width / 2:
            dx += world_width
        if dy > world_height / 2:
            dy -= world_height
        elif dy < -world_height / 2:
            dy += world_height
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < min_enemy_dist:
            min_enemy_dist = dist
            closest_enemy = enemy

    min_ast_dist = float('inf')
    closest_asteroid = None
    for asteroid in env.asteroids:
        dx = asteroid.pos.x - player_pos.x
        dy = asteroid.pos.y - player_pos.y
        if dx > world_width / 2:
            dx -= world_width
        elif dx < -world_width / 2:
            dx += world_width
        if dy > world_height / 2:
            dy -= world_height
        elif dy < -world_height / 2:
            dy += world_height
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < min_ast_dist:
            min_ast_dist = dist
            closest_asteroid = asteroid

    if min_enemy_dist < safe_dist and closest_enemy:
        dx = player_pos.x - closest_enemy.pos.x
        dy = player_pos.y - closest_enemy.pos.y
        if dx > world_width / 2:
            dx -= world_width
        elif dx < -world_width / 2:
            dx += world_width
        if dy > world_height / 2:
            dy -= world_height
        elif dy < -world_height / 2:
            dy += world_height
        desired_dir = pygame.Vector2(dx, dy).normalize()
    elif closest_asteroid:
        dx = closest_asteroid.pos.x - player_pos.x
        dy = closest_asteroid.pos.y - player_pos.y
        if dx > world_width / 2:
            dx -= world_width
        elif dx < -world_width / 2:
            dx += world_width
        if dy > world_height / 2:
            dy -= world_height
        elif dy < -world_height / 2:
            dy += world_height
        desired_dir = pygame.Vector2(dx, dy).normalize()
    else:
        return [0, 0, 0]

    current_heading = pygame.Vector2(1, 0).rotate(-env.player.angle)
    cross = current_heading.x * desired_dir.y - current_heading.y * desired_dir.x
    if cross > 0.1:
        movement = 3
    elif cross < -0.1:
        movement = 4
    else:
        movement = 1

    mining = 1 if (closest_asteroid and 
                  min_ast_dist < mining_radius + closest_asteroid.radius and 
                  min_enemy_dist > 100) else 0

    return [movement, mining, 0]