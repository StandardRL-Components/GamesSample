def policy(env):
    # Always follows the optimal A* path to the exit for maximum reward, using environment's built-in pathfinding.
    # This ensures we move in the correct direction (getting +1 reward) and avoid walls/incorrect moves (-1 reward).
    if env.game_over:
        return [0, 0, 0]
    
    path = env._a_star_path(env.player_pos, env.exit_pos)
    if path and len(path) > 0:
        next_step = path[0]
        dx = next_step[0] - env.player_pos[0]
        dy = next_step[1] - env.player_pos[1]
        if dx == 0 and dy == -1:
            movement = 1
        elif dx == 0 and dy == 1:
            movement = 2
        elif dx == -1 and dy == 0:
            movement = 3
        elif dx == 1 and dy == 0:
            movement = 4
        else:
            movement = 0
    else:
        movement = 0
        
    return [movement, 0, 0]