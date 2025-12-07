def policy(env):
    """
    Maximizes reward by prioritizing risky shots when within 10 pixels for higher points (10 vs 5),
    using safe shots within 25 pixels otherwise. Moves toward closest active target when out of range.
    Conserves ammo by only shooting when in range and targets remain. Always shoots if possible since
    misses have minor penalty (-0.2) vs hits (+1.0 or +0.5), and ammo (15) exceeds targets (10).
    """
    active_targets = [t for t in env.targets if t['active']]
    if not active_targets or env.ammo <= 0:
        return [0, 0, 0]
    
    closest = min(active_targets, key=lambda t: env.crosshair_pos.distance_to(t['pos']))
    dist = env.crosshair_pos.distance_to(closest['pos'])
    
    if dist <= 10:
        return [0, 1, 0]  # Risky shot
    elif dist <= 25:
        return [0, 1, 1]  # Safe shot
    
    dx = closest['pos'].x - env.crosshair_pos.x
    dy = closest['pos'].y - env.crosshair_pos.y
    if abs(dx) > abs(dy):
        move = 4 if dx > 0 else 3
    else:
        move = 2 if dy > 0 else 1
    return [move, 0, 0]