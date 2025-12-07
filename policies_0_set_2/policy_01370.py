def policy(env):
    # Strategy: Prioritize reaching exit while avoiding unnecessary combat. In combat, attack unless health/sanity is critically low then defend.
    if env.current_monster_in_combat is not None:
        # Combat logic: defend if health or sanity is critically low, else attack
        if env.player_health <= 3 or env.player_sanity <= 1:
            return [0, 0, 1]  # Defend
        else:
            return [0, 1, 0]  # Attack
    else:
        # Movement logic: move toward exit using Manhattan distance, prioritizing larger axis differences
        dx = env.exit_pos[0] - env.player_pos[0]
        dy = env.exit_pos[1] - env.player_pos[1]
        if abs(dx) > abs(dy):
            movement = 4 if dx > 0 else 3  # Right/Left
        else:
            movement = 2 if dy > 0 else 1  # Down/Up
        return [movement, 0, 0]