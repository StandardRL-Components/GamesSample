def policy(env):
    """Maximizes reward by prioritizing immediate item collection, then targeting the most efficient item (closest and in direction of treasure) to minimize backtracking. Uses dash when available and either items are within dash reward radius or no items remain and treasure is far, ensuring efficient movement and avoiding penalties."""
    player_pos = env.player_pos
    treasure_pos = env.treasure['pos']
    active_items = [item for item in env.items if item['active']]
    
    # Immediate collection check
    collect = 1 if any(player_pos.distance_to(item['pos']) < env.COLLECT_RADIUS for item in active_items) else 0
    
    # Target selection: closest item considering treasure direction when ties occur
    if active_items:
        # Prefer items that are both close and in direction of treasure
        target = min(active_items, key=lambda item: (
            player_pos.distance_to(item['pos']) + 
            item['pos'].distance_to(treasure_pos) * 0.3  # Weight treasure direction
        ))['pos']
    else:
        target = treasure_pos
    
    # Movement direction with tie-breaking towards treasure
    dx = target.x - player_pos.x
    dy = target.y - player_pos.y
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1
    
    # Strategic dash usage
    dash = 0
    if env.dash_cooldown_timer <= 0 and env.dash_timer <= 0:
        if active_items:
            # Dash if any item within reward radius
            if any(player_pos.distance_to(item['pos']) < env.DASH_REWARD_RADIUS for item in active_items):
                dash = 1
        else:
            # Dash towards treasure if far away
            if player_pos.distance_to(treasure_pos) > 200:
                dash = 1
    
    return [movement, collect, dash]