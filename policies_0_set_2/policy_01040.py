def policy(env):
    # Strategy: Prioritize moving towards the nearest gem while avoiding mines. 
    # Use Euclidean distance to evaluate candidate moves, heavily penalizing mines and rewarding gems.
    # Break ties by preferring movement over staying still to maximize progress under time constraints.
    px, py = env.player_pos
    gems = env.gems
    mines = env.mines
    maze = env.maze
    best_action = 0
    best_score = float('-inf')
    
    if not gems:
        return [0, 0, 0]
    
    for action in range(5):
        if action == 0:
            new_pos = (px, py)
        else:
            dx, dy = 0, 0
            if action == 1 and py > 0 and not maze[px][py]['walls'][0]:
                dy = -1
            elif action == 2 and py < env.MAZE_HEIGHT - 1 and not maze[px][py]['walls'][2]:
                dy = 1
            elif action == 3 and px > 0 and not maze[px][py]['walls'][3]:
                dx = -1
            elif action == 4 and px < env.MAZE_WIDTH - 1 and not maze[px][py]['walls'][1]:
                dx = 1
            else:
                continue
            new_pos = (px + dx, py + dy)
        
        if new_pos in mines:
            score = -10000
        elif new_pos in gems:
            score = 1000
        else:
            min_dist = min((gx - new_pos[0])**2 + (gy - new_pos[1])**2 for gx, gy in gems)
            score = -min_dist
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return [best_action, 0, 0]