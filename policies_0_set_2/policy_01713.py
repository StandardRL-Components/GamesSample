def policy(env):
    """
    Maximizes reward by prioritizing survival through dodging enemy projectiles, 
    collecting powerups for defense/offense, and eliminating enemies when safe.
    Uses internal state for precise positioning and threat assessment.
    """
    import math
    import pygame

    # If game over, return neutral action
    if env.game_over:
        return [0, 0, 0]

    player = env.player
    player_pos = player['pos']
    enemy_bullets = [p for p in env.projectiles if p['owner'] == 'enemy']
    
    # Initialize movement to none
    movement = 0
    fire = 0
    secondary = 0

    # Dodge enemy bullets if any are close and threatening
    if enemy_bullets:
        closest_bullet = min(enemy_bullets, key=lambda b: player_pos.distance_to(b['pos']))
        bullet_dist = player_pos.distance_to(closest_bullet['pos'])
        if bullet_dist < 100:
            # Calculate dodge direction perpendicular to bullet trajectory
            bullet_vel = closest_bullet['vel'].normalize()
            dodge_dir = pygame.math.Vector2(-bullet_vel.y, bullet_vel.x)
            
            # Choose discrete movement direction closest to dodge vector
            directions = [
                (1, pygame.math.Vector2(0, -1)),   # up
                (2, pygame.math.Vector2(0, 1)),    # down
                (3, pygame.math.Vector2(-1, 0)),   # left
                (4, pygame.math.Vector2(1, 0))     # right
            ]
            best_dir = max(directions, key=lambda d: dodge_dir.dot(d[1]))
            movement = best_dir[0]

    # If no immediate threat, prioritize powerup collection or enemy engagement
    if movement == 0:
        if env.powerups and player['health'] < 70:
            # Move toward closest powerup if health is low
            closest_powerup = min(env.powerups, key=lambda p: player_pos.distance_to(p['pos']))
            powerup_dir = (closest_powerup['pos'] - player_pos).normalize()
            directions = [
                (1, pygame.math.Vector2(0, -1)),
                (2, pygame.math.Vector2(0, 1)),
                (3, pygame.math.Vector2(-1, 0)),
                (4, pygame.math.Vector2(1, 0))
            ]
            best_dir = max(directions, key=lambda d: powerup_dir.dot(d[1]))
            movement = best_dir[0]
        elif env.enemies:
            # Move toward closest enemy
            closest_enemy = min(env.enemies, key=lambda e: player_pos.distance_to(e['pos']))
            enemy_dir = (closest_enemy['pos'] - player_pos).normalize()
            directions = [
                (1, pygame.math.Vector2(0, -1)),
                (2, pygame.math.Vector2(0, 1)),
                (3, pygame.math.Vector2(-1, 0)),
                (4, pygame.math.Vector2(1, 0))
            ]
            best_dir = max(directions, key=lambda d: enemy_dir.dot(d[1]))
            movement = best_dir[0]

    # Fire if cooldown is ready and enemies are present
    if player['fire_cooldown'] <= 0 and env.enemies:
        fire = 1

    return [movement, fire, secondary]