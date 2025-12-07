def policy(env):
    # Strategy: Track the lowest fruit and move basket to intercept its predicted landing position.
    # Prioritize catching fruits to maximize score and minimize misses, using horizontal velocity
    # prediction for accuracy when fruit is close to basket. Secondary actions are unused in this game.
    basket_center = env.basket_rect.centerx
    basket_top = env.basket_rect.top
    active_fruits = [f for f in env.fruits if f['pos'][1] < basket_top]
    
    if not active_fruits:
        return [0, 0, 0]
    
    target_fruit = max(active_fruits, key=lambda f: f['pos'][1])
    target_x = target_fruit['pos'][0]
    
    # Predict horizontal movement for fruits close to basket
    if target_fruit['pos'][1] > basket_top - 50:
        time_to_impact = (basket_top - target_fruit['pos'][1]) / target_fruit['vel'][1]
        target_x += target_fruit['vel'][0] * time_to_impact
        target_x = max(target_fruit['radius'], min(env.WIDTH - target_fruit['radius'], target_x))
    
    if abs(target_x - basket_center) < 5:
        return [0, 0, 0]
    return [3 if target_x < basket_center else 4, 0, 0]