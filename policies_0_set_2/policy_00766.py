def policy(env):
    # Strategy: Prioritize slicing low fruits to prevent misses, move towards most urgent fruit (lowest Y),
    # and slice when cursor is near any fruit (edge-triggered) to maximize score and minimize misses.
    if env.game_over:
        return [0, 0, 0]
    
    fruits = env.fruits
    if not fruits:
        return [0, 0, 0]
    
    # Find most urgent fruit (lowest on screen)
    target_fruit = max(fruits, key=lambda f: f['pos'].y)
    
    # Move towards target fruit
    dx = target_fruit['pos'].x - env.cursor_pos.x
    dy = target_fruit['pos'].y - env.cursor_pos.y
    if abs(dx) > abs(dy):
        movement = 4 if dx > 0 else 3
    else:
        movement = 2 if dy > 0 else 1
    
    # Check if near any fruit to slice (edge-triggered)
    near_fruit = any(env.cursor_pos.distance_to(f['pos']) < f['radius'] + 20 for f in fruits)
    slice_action = 1 if (near_fruit and not env.last_space_press) else 0
    
    return [movement, slice_action, 0]