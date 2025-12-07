def policy(env):
    # Strategy: Prioritize shooting when cooldown is ready to eliminate enemies quickly for rewards.
    # Move towards nearest enemy to align for shooting while maintaining safe distance (~200px).
    # Never jump since enemies are ground-based and projectiles are non-threatening in current implementation.
    if env.enemies:
        player_x = env.player['pos'].x
        nearest_enemy = min(env.enemies, key=lambda e: abs(e['pos'].x - player_x))
        dx = nearest_enemy['pos'].x - player_x
        current_facing = env.player['facing_dir']
        
        if dx == 0:
            movement = 4  # Break exact alignment by moving right
        elif (dx < 0 and current_facing != -1) or (dx > 0 and current_facing != 1):
            movement = 3 if dx < 0 else 4  # Turn to face enemy
        else:
            movement = 3 if dx < 0 else 4 if abs(dx) > 200 else 0  # Approach if far, else hold position
    else:
        movement = 0  # No movement if no enemies
        
    shoot_ready = env.player['shoot_cooldown'] <= 0
    return [movement, 0, 1 if shoot_ready and env.enemies else 0]