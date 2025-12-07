
# Generated: 2025-08-28T01:28:49.264048
# Source Brief: brief_04122.md
# Brief Index: 4122

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to turn. Complete 3 laps before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Race against AI opponents on a procedurally generated track. Finish 3 laps in under 60 seconds to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30 # Game logic runs at this rate

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (30, 40, 50)
        self.COLOR_TRACK = (80, 90, 100)
        self.COLOR_LINES = (200, 200, 210)
        self.COLOR_PLAYER = (255, 80, 80)
        self.COLOR_OPPONENTS = [(80, 120, 255), (100, 255, 100), (255, 240, 100)]
        self.COLOR_UI_TEXT = (240, 240, 240)
        self.COLOR_FINISH_LINE_1 = (255, 255, 255)
        self.COLOR_FINISH_LINE_2 = (20, 20, 20)

        # Physics constants
        self.MAX_SPEED = 6.0
        self.ACCELERATION = 0.15
        self.BRAKE_DECELERATION = 0.3
        self.FRICTION = 0.03
        self.TURN_SPEED = 0.08 # Radians per frame
        
        # Game constants
        self.NUM_OPPONENTS = 3
        self.LAPS_TO_WIN = 3
        self.GAME_DURATION_SECONDS = 60
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        
        # Initialize state variables to be populated in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_outcome = None
        self.timer = None
        self.track_center_line = None
        self.track_poly = None
        self.checkpoints = None
        self.player = None
        self.opponents = None
        self.particles = None
        
        # Initialize state variables
        self.reset()

        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        self.timer = self.MAX_STEPS

        # Procedurally generate track
        self._generate_track()

        # Initialize cars
        self._init_cars()
        
        # Particles for effects
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            
            # Update game state
            self.steps += 1
            self.timer -= 1
            
            # Update player and get immediate reward
            progress_reward = self._update_player(movement)
            reward += progress_reward

            # Update opponents and particles
            self._update_opponents()
            self._update_particles()
        
        # Check for termination and calculate terminal reward
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            terminal_reward = self._calculate_terminal_reward()
            reward += terminal_reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _generate_track(self):
        """Generates a closed-loop track with checkpoints."""
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        num_points = 12
        min_radius, max_radius = 100, 160
        angle_variation = 0.4 # Radians
        radius_variation = 30
        
        self.track_center_line = []
        angle_step = (2 * math.pi) / num_points
        
        for i in range(num_points):
            base_angle = i * angle_step
            angle = base_angle + self.np_random.uniform(-angle_variation, angle_variation)
            radius = self.np_random.uniform(min_radius, max_radius)
            
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.track_center_line.append(pygame.Vector2(x, y))

        # Create a smooth path and checkpoints from the points
        self.checkpoints = []
        num_checkpoints_per_segment = 8
        for i in range(num_points):
            p1 = self.track_center_line[i]
            p2 = self.track_center_line[(i + 1) % num_points]
            for j in range(num_checkpoints_per_segment):
                t = j / num_checkpoints_per_segment
                pt = p1.lerp(p2, t)
                self.checkpoints.append(pt)

        # Create the visible track polygon
        track_width = 40
        self.track_poly = []
        outer_edge = []
        inner_edge = []

        for i in range(len(self.checkpoints)):
            p1 = self.checkpoints[i]
            p2 = self.checkpoints[(i + 1) % len(self.checkpoints)]
            
            angle = math.atan2(p2.y - p1.y, p2.x - p1.x) + math.pi / 2
            
            outer_point = p1 + pygame.Vector2(math.cos(angle), math.sin(angle)) * track_width / 2
            inner_point = p1 - pygame.Vector2(math.cos(angle), math.sin(angle)) * track_width / 2
            
            outer_edge.append(outer_point)
            inner_edge.append(inner_point)

        self.track_poly = outer_edge + inner_edge[::-1]

    def _init_cars(self):
        """Initializes player and opponent car states."""
        start_pos = self.checkpoints[0]
        next_pos = self.checkpoints[1]
        start_angle = math.atan2(next_pos.y - start_pos.y, next_pos.x - start_pos.x)
        
        # Player car
        self.player = {
            "pos": pygame.Vector2(start_pos.x, start_pos.y),
            "angle": start_angle,
            "speed": 0.0,
            "lap": 0,
            "checkpoint": 0,
            "width": 12, "height": 22
        }

        # Opponent cars
        self.opponents = []
        for i in range(self.NUM_OPPONENTS):
            offset_angle = start_angle + math.pi / 2
            offset_dist = (i - (self.NUM_OPPONENTS - 1) / 2) * 20
            pos = start_pos + pygame.Vector2(math.cos(offset_angle), math.sin(offset_angle)) * offset_dist
            pos -= pygame.Vector2(math.cos(start_angle), math.sin(start_angle)) * (i + 1) * 15 # Stagger start
            
            self.opponents.append({
                "pos": pos,
                "angle": start_angle,
                "speed": self.np_random.uniform(2.5, 3.5),
                "lap": 0,
                "checkpoint": 0,
                "target_checkpoint": 1,
                "width": 12, "height": 22,
                "color": self.COLOR_OPPONENTS[i % len(self.COLOR_OPPONENTS)],
                "max_speed": self.np_random.uniform(3.8, 4.5)
            })

    def _update_player(self, movement):
        """Updates player car based on action."""
        # --- Steering ---
        if self.player["speed"] > 0.1: # Can only turn while moving
            if movement == 3: # Left
                self.player["angle"] -= self.TURN_SPEED
            elif movement == 4: # Right
                self.player["angle"] += self.TURN_SPEED
        
        # --- Acceleration & Braking ---
        if movement == 1: # Up
            self.player["speed"] += self.ACCELERATION
            # sfx: engine_accelerate
            if self.np_random.random() < 0.5:
                self._create_exhaust_particle(self.player)
        elif movement == 2: # Down
            self.player["speed"] -= self.BRAKE_DECELERATION
            # sfx: brake_squeal
        else: # No-op
            self.player["speed"] *= (1 - self.FRICTION)

        self.player["speed"] = max(0, min(self.MAX_SPEED, self.player["speed"]))

        # --- Update Position ---
        velocity = pygame.Vector2(math.cos(self.player["angle"]), math.sin(self.player["angle"])) * self.player["speed"]
        self.player["pos"] += velocity

        # --- Wall Collision ---
        # Simple check: if car center is outside track polygon, it's a collision
        if not self._is_point_in_polygon(self.player["pos"], self.track_poly):
            self.player["pos"] -= velocity * 1.5 # Revert movement
            self.player["speed"] *= 0.1 # Drastic speed loss
            # sfx: car_crash
            return -0.1 # Collision penalty
        
        # --- Checkpoint & Lap Logic ---
        reward = 0
        next_checkpoint_idx = (self.player["checkpoint"] + 1) % len(self.checkpoints)
        dist_to_next = self.player["pos"].distance_to(self.checkpoints[next_checkpoint_idx])
        
        if dist_to_next < 25: # Checkpoint passed
            self.player["checkpoint"] = next_checkpoint_idx
            reward += 0.1 # Progress reward
            # sfx: checkpoint_ding

            if self.player["checkpoint"] == 0: # Lap completed
                self.player["lap"] += 1
                reward += 1.0
                # sfx: lap_complete
        
        return reward

    def _update_opponents(self):
        """Updates AI opponent cars."""
        for opp in self.opponents:
            # --- AI Steering ---
            target_pos = self.checkpoints[opp["target_checkpoint"]]
            target_vec = target_pos - opp["pos"]
            target_angle = math.atan2(target_vec.y, target_vec.x)

            # Normalize angles
            angle_diff = (target_angle - opp["angle"] + math.pi) % (2 * math.pi) - math.pi
            
            # Steer towards target
            turn_direction = 0
            if angle_diff > 0.05:
                turn_direction = 1
            elif angle_diff < -0.05:
                turn_direction = -1

            opp["angle"] += turn_direction * self.TURN_SPEED * 0.8 # Slightly worse turning
            
            # --- AI Speed Control ---
            # Slow down for sharp turns
            if abs(angle_diff) > 0.8: # Radians
                opp["speed"] *= 0.95
            else:
                opp["speed"] = min(opp["max_speed"], opp["speed"] + self.ACCELERATION * 0.7)
            
            opp["speed"] *= (1 - self.FRICTION)

            # --- Update Position ---
            velocity = pygame.Vector2(math.cos(opp["angle"]), math.sin(opp["angle"])) * opp["speed"]
            opp["pos"] += velocity

            # --- AI Collision (simplified) ---
            if not self._is_point_in_polygon(opp["pos"], self.track_poly):
                opp["pos"] -= velocity
                opp["speed"] *= 0.5

            # --- AI Checkpoint Logic ---
            dist_to_target = opp["pos"].distance_to(target_pos)
            if dist_to_target < 30:
                opp["target_checkpoint"] = (opp["target_checkpoint"] + 1) % len(self.checkpoints)
                opp["checkpoint"] = (opp["checkpoint"] + 1) % len(self.checkpoints)
                if opp["checkpoint"] == 0:
                    opp["lap"] += 1

    def _update_particles(self):
        """Updates position and lifetime of particles."""
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1

    def _create_exhaust_particle(self, car):
        angle = car["angle"] + math.pi + self.np_random.uniform(-0.2, 0.2)
        speed = self.np_random.uniform(1, 2)
        pos = car["pos"] - pygame.Vector2(math.cos(car["angle"]), math.sin(car["angle"])) * car["height"] / 2
        vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.particles.append({"pos": pos, "vel": vel, "life": 10, "color": (200, 200, 200)})

    def _check_termination(self):
        """Checks if the episode should end."""
        if self.player["lap"] >= self.LAPS_TO_WIN:
            self.game_outcome = "FINISH!"
            return True
        if self.timer <= 0:
            self.game_outcome = "TIME UP"
            return True
        return False

    def _calculate_terminal_reward(self):
        """Calculates reward upon episode termination."""
        if self.player["lap"] >= self.LAPS_TO_WIN:
            rankings = self._get_rankings()
            player_rank = rankings.index("Player") + 1
            if player_rank == 1: return 100 # 1st place
            if player_rank == 2: return 50  # 2nd place
            if player_rank == 3: return 25  # 3rd place
            return 10 # Finished
        return -10 # Ran out of time

    def _get_rankings(self):
        """Returns a sorted list of car names by race progress."""
        all_cars = [{"name": "Player", "lap": self.player["lap"], "checkpoint": self.player["checkpoint"]}]
        for i, opp in enumerate(self.opponents):
            all_cars.append({"name": f"Opponent {i+1}", "lap": opp["lap"], "checkpoint": opp["checkpoint"]})
        
        # Sort by lap (desc), then by checkpoint (desc)
        all_cars.sort(key=lambda c: (c["lap"], c["checkpoint"]), reverse=True)
        return [c["name"] for c in all_cars]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_track()
        self._render_particles()
        self._render_cars()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lap": self.player["lap"],
            "time_left": self.timer / self.FPS,
        }

    def _render_track(self):
        # Draw main track surface
        pygame.gfxdraw.filled_polygon(self.screen, self.track_poly, self.COLOR_TRACK)
        pygame.gfxdraw.aapolygon(self.screen, self.track_poly, self.COLOR_TRACK)

        # Draw track outline for clarity
        pygame.draw.aalines(self.screen, self.COLOR_LINES, True, self.track_poly, 1)

        # Draw finish line
        start_pos = self.checkpoints[0]
        prev_pos = self.checkpoints[-1]
        angle = math.atan2(start_pos.y - prev_pos.y, start_pos.x - prev_pos.x)
        perp_angle = angle + math.pi / 2
        
        line_width = 40
        num_checks = 10
        check_size = line_width / num_checks
        
        for i in range(num_checks):
            dist = -line_width / 2 + i * check_size
            p1 = start_pos + pygame.Vector2(math.cos(perp_angle), math.sin(perp_angle)) * dist
            p2 = p1 + pygame.Vector2(math.cos(perp_angle), math.sin(perp_angle)) * check_size
            p3 = p2 + pygame.Vector2(math.cos(angle), math.sin(angle)) * 5
            p4 = p1 + pygame.Vector2(math.cos(angle), math.sin(angle)) * 5
            
            color = self.COLOR_FINISH_LINE_1 if i % 2 == 0 else self.COLOR_FINISH_LINE_2
            pygame.draw.polygon(self.screen, color, [p1, p2, p3, p4])

    def _render_particles(self):
        for p in self.particles:
            size = max(0, p["life"] / 5)
            pygame.draw.circle(self.screen, p["color"], p["pos"], size)

    def _render_cars(self):
        all_cars = [{"car": opp, "color": opp["color"]} for opp in self.opponents]
        all_cars.append({"car": self.player, "color": self.COLOR_PLAYER})
        
        for car_data in all_cars:
            car = car_data["car"]
            car_surf = pygame.Surface((car["height"], car["width"]), pygame.SRCALPHA)
            pygame.draw.rect(car_surf, car_data["color"], (0, 0, car["height"]-2, car["width"]), border_radius=3)
            # Headlights
            pygame.draw.rect(car_surf, (255, 255, 0), (car["height"]-4, 2, 4, 3))
            pygame.draw.rect(car_surf, (255, 255, 0), (car["height"]-4, car["width"]-5, 4, 3))

            rotated_surf = pygame.transform.rotate(car_surf, -math.degrees(car["angle"]))
            rect = rotated_surf.get_rect(center=car["pos"])
            self.screen.blit(rotated_surf, rect)

    def _render_ui(self):
        # Lap counter
        lap_text = self.font_small.render(f"LAP: {min(self.player['lap'] + 1, self.LAPS_TO_WIN)}/{self.LAPS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_text, (10, 10))

        # Timer
        time_left = max(0, self.timer / self.FPS)
        timer_color = (255, 100, 100) if time_left < 10 else self.COLOR_UI_TEXT
        timer_text = self.font_small.render(f"TIME: {time_left:.2f}", True, timer_color)
        self.screen.blit(timer_text, (self.WIDTH - timer_text.get_width() - 10, 10))

        # Position
        try:
            rankings = self._get_rankings()
            player_rank = rankings.index("Player") + 1
            pos_text = self.font_small.render(f"POS: {player_rank}/{self.NUM_OPPONENTS + 1}", True, self.COLOR_UI_TEXT)
            self.screen.blit(pos_text, (self.WIDTH // 2 - pos_text.get_width() // 2, 10))
        except (ValueError, IndexError):
            pass # Don't render if rankings are not ready

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text = self.font_large.render(self.game_outcome, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
    def _is_point_in_polygon(self, point, polygon):
        """Ray-casting algorithm to check if a point is inside a polygon."""
        x, y = point.x, point.y
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- To display the game in a window ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Racer")
    
    obs, info = env.reset()
    done = False
    
    # Mapping keyboard keys to actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    running = True
    while running:
        movement = 0 # No-op
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4

        # The brief doesn't use space/shift, but the action space includes them
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score'] + reward}")
            pygame.time.wait(2000) # Pause for 2 seconds
            obs, info = env.reset()

        env.clock.tick(env.FPS)
        
    env.close()