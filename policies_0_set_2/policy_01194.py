def policy(env):
    # Strategy: Prioritize slicing fruits while avoiding bombs. Move towards closest fruit,
    # only slice when positioned over fruit (not bomb). Avoid bombs by moving away when too close.
    if env.game_over:
        return [0, 0, 0]
    
    cursor = env.cursor_pos
    fruits = env.fruits
    bombs = env.bombs
    
    # Check current collisions
    on_bomb = any(cursor.distance_to(b['pos']) < b['radius'] for b in bombs)
    on_fruit = any(cursor.distance_to(f['pos']) < f['radius'] for f in fruits)
    
    if on_bomb:
        # Move away from bomb
        bomb = next(b for b in bombs if cursor.distance_to(b['pos']) < b['radius'])
        dx = cursor.x - bomb['pos'].x
        dy = cursor.y - bomb['pos'].y
        if abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 2 if dy > 0 else 1
        return [a0, 0, 0]
    
    if on_fruit:
        # Slice while staying in position
        return [0, 1, 0]
    
    if fruits:
        # Move towards closest fruit
        closest = min(fruits, key=lambda f: cursor.distance_to(f['pos']))
        dx = closest['pos'].x - cursor.x
        dy = closest['pos'].y - cursor.y
        if abs(dx) > abs(dy):
            a0 = 4 if dx > 0 else 3
        else:
            a0 = 2 if dy > 0 else 1
        return [a0, 0, 0]
    
    if bombs:
        # Avoid closest bomb if nearby
        closest_bomb = min(bombs, key=lambda b: cursor.distance_to(b['pos']))
        if cursor.distance_to(closest_bomb['pos']) < 50:
            dx = cursor.x - closest_bomb['pos'].x
            dy = cursor.y - closest_bomb['pos'].y
            if abs(dx) > abs(dy):
                a0 = 4 if dx > 0 else 3
            else:
                a0 = 2 if dy > 0 else 1
            return [a0, 0, 0]
    
    return [0, 0, 0]