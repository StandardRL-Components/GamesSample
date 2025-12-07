def policy(env):
    # Strategy: Maximize immediate adjacency bonuses when placing ingredients, prioritize high-interaction pairs.
    # In browsing mode, select ingredient with highest potential adjacency score based on current grid.
    # In placing mode, move to optimal empty cell for maximum bonus, then place immediately.
    # Break ties by preferring center positions for better future placement opportunities.
    
    def calc_placement_score(x, y, ing_idx):
        score = 0
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < env.GRID_SIZE and 0 <= ny < env.GRID_SIZE:
                neighbor_idx = env.cooking_grid[ny][nx]
                if neighbor_idx != -1:
                    score += env.interaction_matrix[ing_idx][neighbor_idx]
                    score += env.interaction_matrix[neighbor_idx][ing_idx]
        return score

    if env.held_ingredient_idx is not None:
        current_pos = env.cursor_pos
        best_score = -float('inf')
        best_pos = None
        for y in range(env.GRID_SIZE):
            for x in range(env.GRID_SIZE):
                if env.cooking_grid[y][x] == -1:
                    score = calc_placement_score(x, y, env.held_ingredient_idx)
                    if score > best_score or (score == best_score and (x,y) < (best_pos[0], best_pos[1])):
                        best_score = score
                        best_pos = (x, y)
        if best_pos == tuple(current_pos):
            return [0, 1, 0]
        dx = best_pos[0] - current_pos[0]
        dy = best_pos[1] - current_pos[1]
        if dx > 0:
            return [4, 0, 0]
        elif dx < 0:
            return [3, 0, 0]
        elif dy > 0:
            return [2, 0, 0]
        else:
            return [1, 0, 0]
    else:
        existing_ingredients = set()
        for y in range(env.GRID_SIZE):
            for x in range(env.GRID_SIZE):
                if env.cooking_grid[y][x] != -1:
                    existing_ingredients.add(env.cooking_grid[y][x])
        best_ingredient = None
        best_potential = -float('inf')
        for i in range(env.NUM_INGREDIENTS):
            if env.ingredient_counts[i] > 0:
                if existing_ingredients:
                    potential = max(env.interaction_matrix[i][j] + env.interaction_matrix[j][i] for j in existing_ingredients)
                else:
                    potential = sum(env.interaction_matrix[i]) / len(env.interaction_matrix[i])
                if potential > best_potential:
                    best_potential = potential
                    best_ingredient = i
        current = env.selected_ingredient_idx
        if current == best_ingredient:
            return [0, 1, 0]
        n = env.NUM_INGREDIENTS
        clockwise = (best_ingredient - current) % n
        counter_clockwise = (current - best_ingredient) % n
        if clockwise <= counter_clockwise:
            return [4, 0, 0]
        else:
            return [3, 0, 0]