def policy(env):
    # Strategy: Place towers in all available zones to maximize defense. Towers auto-target enemies,
    # so covering all strategic positions ensures maximum enemy elimination and wave survival rewards.
    # We prioritize building during wave prep time and use a fixed zone order to avoid oscillation.
    available_zones = [zone for zone in env.tower_zones if not any(t['pos'] == list(zone) for t in env.towers)]
    if not available_zones:
        return [0, 0, 0]
    sorted_zones = sorted(available_zones, key=lambda z: (z[1], z[0]))
    target = sorted_zones[0]
    cx, cy = env.cursor_pos
    tx, ty = target
    if cx == tx and cy == ty:
        return [0, 1, 0]
    dx = (tx - cx) % env.GRID_WIDTH
    if dx > env.GRID_WIDTH // 2:
        dx -= env.GRID_WIDTH
    dy = (ty - cy) % env.GRID_HEIGHT
    if dy > env.GRID_HEIGHT // 2:
        dy -= env.GRID_HEIGHT
    if abs(dx) > abs(dy):
        return [4 if dx > 0 else 3, 0, 0]
    else:
        return [2 if dy > 0 else 1, 0, 0]