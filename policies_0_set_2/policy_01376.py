def policy(env):
    """
    Maximizes reward by tracking the lowest fruit above the basket and moving to intercept it.
    Avoids unnecessary movement penalties by only acting when fruits are in the lower screen.
    Prioritizes catching fruits (immediate reward) over other actions.
    """
    basket_x = env.basket_pos.x
    basket_y = env.basket_pos.y
    basket_width = env.BASKET_WIDTH
    fruit_radius = env.FRUIT_RADIUS
    
    # Find fruits in lower 80% of screen and above basket
    lower_threshold = 0.2 * env.HEIGHT
    candidate_fruits = []
    for fruit in env.fruits:
        fruit_y = fruit['pos'].y
        if fruit_y < lower_threshold:
            continue
        if fruit_y > basket_y + env.BASKET_HEIGHT/2 + fruit_radius:
            continue
        candidate_fruits.append(fruit)
    
    if not candidate_fruits:
        return [0, 0, 0]
    
    # Target the lowest fruit
    most_urgent_fruit = max(candidate_fruits, key=lambda f: f['pos'].y)
    
    # Check if already aligned
    basket_left = basket_x - basket_width/2
    basket_right = basket_x + basket_width/2
    fruit_left = most_urgent_fruit['pos'].x - fruit_radius
    fruit_right = most_urgent_fruit['pos'].x + fruit_radius
    
    if basket_left <= fruit_right and basket_right >= fruit_left:
        return [0, 0, 0]
    
    # Move toward target fruit
    if most_urgent_fruit['pos'].x < basket_x:
        return [3, 0, 0]
    else:
        return [4, 0, 0]