def policy(env):
    # Strategy: Track the lowest fruit (most urgent) and move baskets to catch good fruits (green/red) or avoid blue ones.
    # Prioritize by vertical position (lowest first) and horizontal alignment with basket group center.
    if env.game_over:
        return [0, 0, 0]
    
    fruits_above = [f for f in env.fruits if f['pos'][1] < env.BASKET_Y]
    if not fruits_above:
        return [0, 0, 0]
    
    urgent_fruit = min(fruits_above, key=lambda f: f['pos'][1])
    group_center = env.baskets_x + env.total_baskets_width / 2
    
    if urgent_fruit['type'] in ['green', 'red']:
        diff = group_center - urgent_fruit['pos'][0]
        if diff < -5:
            return [4, 0, 0]
        elif diff > 5:
            return [3, 0, 0]
        else:
            return [0, 0, 0]
    else:  # blue fruit
        if env.baskets_x <= urgent_fruit['pos'][0] <= env.baskets_x + env.total_baskets_width:
            if group_center < urgent_fruit['pos'][0]:
                return [3, 0, 0]
            else:
                return [4, 0, 0]
        return [0, 0, 0]