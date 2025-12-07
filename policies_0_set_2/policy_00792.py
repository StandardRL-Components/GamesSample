def policy(env):
    # Strategy: Prioritize catching the lowest fruit to maximize score and maintain combo, while avoiding bombs that are close and aligned with the basket.
    # Horizontal movement is sufficient since fruits fall vertically; vertical movement is avoided to minimize risk and maintain position for future items.
    avoid_threshold = 50  # vertical distance to consider bombs dangerous
    current_x = env.player_pos[0]
    current_y = env.player_pos[1]
    
    # Avoid bombs that are close and aligned with the basket
    for item in env.items:
        if item['type'] == 'bomb':
            bomb_x, bomb_y = item['pos']
            if bomb_y < current_y and current_y - bomb_y < avoid_threshold:
                if abs(bomb_x - current_x) < env.PLAYER_WIDTH // 2:
                    if bomb_x < current_x:
                        return [4, 0, 0]  # move right
                    else:
                        return [3, 0, 0]  # move left
    
    # Target the lowest fruit above the basket
    lowest_fruit = None
    max_y = -1
    for item in env.items:
        if item['type'] != 'bomb' and item['pos'][1] < current_y:
            if item['pos'][1] > max_y:
                max_y = item['pos'][1]
                lowest_fruit = item
    
    if lowest_fruit is not None:
        fruit_x = lowest_fruit['pos'][0]
        if fruit_x < current_x - 5:
            return [3, 0, 0]  # move left
        elif fruit_x > current_x + 5:
            return [4, 0, 0]  # move right
    
    return [0, 0, 0]  # no movement needed