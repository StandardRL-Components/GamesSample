def policy(env):
    # Strategy: Build track segments to the right while maintaining a safe height above terrain.
    # Prioritize extending rightwards to progress towards finish, adjusting vertically to avoid crashes.
    # Use boost for right moves when aligned with terrain to cover more ground efficiently.
    if env.game_over:
        return [0, 0, 0]
    
    current_end = env.track_points[-1]
    if current_end.x >= env.FINISH_X:
        return [0, 0, 0]
    
    target_x = current_end.x + env.DRAW_LENGTH
    target_x = min(target_x, env.FINISH_X)
    desired_y = env._get_terrain_y(target_x) - 20
    
    current_y = current_end.y
    if current_y > desired_y + 10:
        return [2, 0, 0]
    elif current_y < desired_y - 10:
        return [1, 0, 0]
    else:
        boost = 1 if current_end.x + env.DRAW_LENGTH_BOOST <= env.FINISH_X else 0
        return [4, boost, 0]