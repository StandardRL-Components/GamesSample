def policy(env):
    """
    Maximizes reward by prioritizing immediate harvests for quick coin conversion,
    then planting to ensure future harvests, using efficient movement to minimize turns.
    Targets closest harvestable crops first to reduce movement waste.
    """
    current = env.selected_target
    harvested = env.harvested_crops
    plots = env.plots

    # Sell harvested crops if available (highest priority)
    if harvested > 0:
        if current == 10:
            return [0, 1, 0]  # Sell at market
        return [2, 0, 0]  # Move to market

    # Find closest harvest-ready plot
    ready_plots = [i for i in range(10) if plots[i][0] == 2]
    if ready_plots:
        # Calculate distances from current position
        if current == 10:
            distances = [1 + abs(4 - i) for i in ready_plots]  # Market to plot via plot 4
        else:
            distances = [abs(current - i) for i in ready_plots]
        target = ready_plots[distances.index(min(distances))]
        
        if current == target:
            return [0, 1, 0]  # Harvest ready crop
        elif current == 10:
            return [1, 0, 0]  # Move up from market to plot 4
        elif current < target:
            return [4, 0, 0]  # Move right
        else:
            return [3, 0, 0]  # Move left

    # Find closest empty plot for planting
    empty_plots = [i for i in range(10) if plots[i][0] == 0]
    if empty_plots:
        if current == 10:
            distances = [1 + abs(4 - i) for i in empty_plots]
        else:
            distances = [abs(current - i) for i in empty_plots]
        target = empty_plots[distances.index(min(distances))]
        
        if current == target:
            return [0, 1, 0]  # Plant seed
        elif current == 10:
            return [1, 0, 0]  # Move up from market
        elif current < target:
            return [4, 0, 0]  # Move right
        else:
            return [3, 0, 0]  # Move left

    return [0, 0, 0]  # Wait if no actions available