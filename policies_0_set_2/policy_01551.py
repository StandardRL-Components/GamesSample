def policy(env):
    """
    Maximizes reward by mining asteroids while avoiding collisions. Prioritizes aligning with and mining the closest asteroid within range (300 units) to gain ore. Avoids collisions by braking when asteroids are close and in front, or turning away when very near. Uses thrust to maintain mobility when no immediate actions are needed.
    """
    import math

    if not env.asteroids:
        return [1, 0, 0]  # Thrust forward if no asteroids

    player_pos = env.player_pos
    player_angle = env.player_angle
    closest_asteroid = None
    min_dist = float('inf')
    
    for asteroid in env.asteroids:
        dx = asteroid["pos"][0] - player_pos[0]
        dy = asteroid["pos"][1] - player_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < min_dist:
            min_dist = dist
            closest_asteroid = asteroid
            dx_closest = dx
            dy_closest = dy

    if closest_asteroid is None:
        return [1, 0, 0]  # Thrust forward if no valid asteroid

    # Calculate desired angle to asteroid
    desired_angle = math.degrees(math.atan2(dy_closest, dx_closest))
    angle_diff = desired_angle - player_angle
    angle_diff = (angle_diff + 180) % 360 - 180  # Normalize to [-180, 180]

    # Avoid collision: brake if asteroid is close and in front
    if min_dist < 100 and abs(angle_diff) < 90:
        return [2, 0, 0]  # Brake

    # Avoid very close asteroids by turning away
    if min_dist < 50:
        if angle_diff > 0:
            return [4, 0, 0]  # Turn right
        else:
            return [3, 0, 0]  # Turn left

    # Mine if aligned and within range
    if min_dist <= 300 and abs(angle_diff) < 10:
        return [0, 1, 0]  # Fire laser

    # Turn towards asteroid
    if angle_diff > 0:
        return [4, 0, 0]  # Turn right
    else:
        return [3, 0, 0]  # Turn left