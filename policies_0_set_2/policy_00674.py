def policy(env):
    # Strategy: Read the current car position and track center from env's public state to steer towards the track center.
    # This maximizes reward by keeping the car centered on the track, avoiding penalties and earning stay-on-track rewards.
    car_y = env.car_y
    track_index = int(env.track_scroll + env.CAR_X_POS)
    if track_index < len(env.track_points):
        track_center = env.track_points[track_index]
    else:
        track_center = env.track_points[-1]
    
    if car_y < track_center - 2:
        action0 = 2  # down
    elif car_y > track_center + 2:
        action0 = 1  # up
    else:
        action0 = 0  # none
        
    return [action0, 0, 0]