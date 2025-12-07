def policy(env):
    # Strategy: Prioritize attacking adjacent enemies (shift for weakest) to maximize damage and rewards.
    # If no enemy in range, move towards nearest enemy to set up future attacks while avoiding unnecessary steps.
    robot_pos = env.robot_pos
    alive_enemies = [e for e in env.enemies if e["alive"]]
    
    # Check for adjacent enemies (attack range=1)
    adjacent_enemies = []
    for enemy in alive_enemies:
        dx = abs(robot_pos[0] - enemy["pos"][0])
        dy = abs(robot_pos[1] - enemy["pos"][1])
        if dx + dy <= 1:  # Manhattan distance <=1
            adjacent_enemies.append(enemy)
    
    if adjacent_enemies:
        # Attack weakest adjacent enemy using shift (a2=1)
        return [0, 0, 1]
    
    # Move toward nearest enemy
    if alive_enemies:
        nearest = min(alive_enemies, key=lambda e: abs(robot_pos[0]-e["pos"][0]) + abs(robot_pos[1]-e["pos"][1]))
        dx = nearest["pos"][0] - robot_pos[0]
        dy = nearest["pos"][1] - robot_pos[1]
        
        # Prefer dominant axis movement, break ties horizontally
        if abs(dx) > abs(dy):
            move = 4 if dx > 0 else 3
        else:
            move = 2 if dy > 0 else 1
        return [move, 0, 0]
    
    return [0, 0, 0]  # Default if no enemies