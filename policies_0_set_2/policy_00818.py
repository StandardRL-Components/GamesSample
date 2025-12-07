def policy(env):
    # Strategy: Build slow towers (type2) in all zones for maximum coverage and damage.
    # Prioritize building in current zone if empty, else move right to next empty zone.
    # Avoids penalties by not building on occupied zones and ensures all zones are utilized.
    empty_zones = [i for i in range(4) if env.placed_towers[i] is None]
    if not empty_zones:
        return [0, 0, 0]
    current_zone = env.selected_zone_idx
    if env.placed_towers[current_zone] is None:
        return [0, 0, 1]
    else:
        return [4, 0, 0]