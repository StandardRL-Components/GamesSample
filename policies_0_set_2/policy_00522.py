def policy(env):
    # Strategy: Focus on catching the lowest (most imminent) star by moving the basket to align its center with the star's x-position. 
    # Avoid unnecessary movement to prevent oscillation and maximize catch efficiency. Secondary actions are unused in this environment.
    if env.game_over or not env.stars:
        return [0, 0, 0]
    
    lowest_star = max(env.stars, key=lambda s: s['y'])
    basket_center = env.basket_x + env.basket_width / 2
    star_x = lowest_star['x']
    
    if abs(star_x - basket_center) <= 5:
        return [0, 0, 0]
    elif star_x < basket_center:
        return [3, 0, 0]
    else:
        return [4, 0, 0]