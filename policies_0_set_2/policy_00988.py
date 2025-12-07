def policy(env):
    """
    Navigates alien towards spaceship while avoiding lasers. Prioritizes:
    1. Reaching the goal (high reward)
    2. Avoiding collisions (high penalty)
    3. Reducing Manhattan distance to goal
    Uses tie-breaking: right > up > down > left > no-op to prevent oscillation.
    """
    if hasattr(env, 'game_over_message') and env.game_over_message:
        return [0, 0, 0]
    
    alien_pos = env.alien_pos
    spaceship_pos = env.spaceship_pos
    lasers = env.lasers
    laser_speed = env.laser_speed
    grid_width, grid_height = env.GRID_WIDTH, env.GRID_HEIGHT
    
    actions = [0, 1, 2, 3, 4]
    tie_breaking = {4: 0, 1: 1, 2: 2, 3: 3, 0: 4}
    best_action = 0
    best_score = -float('inf')
    
    for a in actions:
        x, y = alien_pos
        if a == 1: y = max(0, y-1)
        elif a == 2: y = min(grid_height-1, y+1)
        elif a == 3: x = max(0, x-1)
        elif a == 4: x = min(grid_width-1, x+1)
        next_pos = [x, y]
        
        if next_pos == spaceship_pos:
            score = 1000
        else:
            collision = False
            for laser in lasers:
                new_pos = laser['pos'] + laser['dir'] * laser_speed
                if laser['type'] == 'v':
                    if not (0 <= new_pos < grid_width):
                        new_pos = max(0, min(grid_width-0.01, new_pos))
                    if int(new_pos) == next_pos[0]:
                        collision = True
                        break
                else:
                    if not (0 <= new_pos < grid_height):
                        new_pos = max(0, min(grid_height-0.01, new_pos))
                    if int(new_pos) == next_pos[1]:
                        collision = True
                        break
            if collision:
                score = -1000
            else:
                dist = abs(next_pos[0]-spaceship_pos[0]) + abs(next_pos[1]-spaceship_pos[1])
                score = -dist
        
        if score > best_score or (score == best_score and tie_breaking[a] < tie_breaking[best_action]):
            best_score = score
            best_action = a
            
    return [best_action, 0, 0]