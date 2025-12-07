def policy(env):
    # Strategy: Prioritize building towers in fixed strategic order (plots 2,4,1,0,3,5,6,7) to cover path choke points.
    # Use Machine Guns early/when poor for farming, Cannons later/when rich for burst. Move cursor efficiently to target plots.
    if env.game_over or env.game_won:
        return [0, 0, 0]
    n = 8
    empty_plots = [i for i in range(n) if not any(tower['pos'] == env.tower_plots[i] for tower in env.towers)]
    if empty_plots and env.resources >= 30:
        if env.wave_number <= 3 or env.resources < 100:
            desired_tower_type = 0
        else:
            desired_tower_type = 1
        if desired_tower_type == 1 and env.resources < 75:
            desired_tower_type = 0
        plot_order = [2, 4, 1, 0, 3, 5, 6, 7]
        target_plot = next((idx for idx in plot_order if idx in empty_plots), None)
        if target_plot is None:
            return [0, 0, 0]
        current_plot = env.cursor_index
        if current_plot == target_plot:
            if env.selected_tower_type_idx == desired_tower_type:
                return [0, 1, 0]
            else:
                return [0, 0, 1]
        else:
            def circular_dist(a, b):
                d1 = (b - a) % n
                d2 = (a - b) % n
                return min(d1, d2)
            best_move = 0
            best_dist = circular_dist(current_plot, target_plot)
            for move in [4, 3, 2, 1]:
                if move == 4:
                    new_plot = (current_plot + 1) % n
                elif move == 3:
                    new_plot = (current_plot - 1) % n
                elif move == 2:
                    new_plot = (current_plot + 2) % n
                else:
                    new_plot = (current_plot - 2) % n
                new_dist = circular_dist(new_plot, target_plot)
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_move = move
            return [best_move, 0, 0]
    else:
        return [0, 0, 0]