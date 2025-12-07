
# Generated: 2025-08-28T00:03:05.418624
# Source Brief: brief_03670.md
# Brief Index: 3670

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑ to accelerate, ↓ to brake, ←→ to steer. Hold space to drift and build boost, hold shift to use boost."
    )

    game_description = (
        "Race against the clock in a top-down kart racer. Drift through corners to build your boost meter, then unleash it for a burst of speed. Complete 3 laps before time runs out!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        # Colors
        self.COLOR_BG = (30, 30, 40)
        self.COLOR_TRACK = (80, 80, 90)
        self.COLOR_WALL = (150, 150, 160)
        self.COLOR_FINISH_LINE = (255, 255, 255)
        self.COLOR_PLAYER = (255, 50, 50)
        self.COLOR_PLAYER_GLOW = (255, 100, 100, 100)
        self.COLOR_COIN = (255, 223, 0)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_BOOST_BAR = (0, 150, 255)
        self.COLOR_BOOST_PARTICLE = (100, 180, 255)

        # Physics constants
        self.ACCELERATION = 0.1
        self.BRAKE_POWER = 0.2
        self.FRICTION = 0.985
        self.TURN_SPEED = 0.05
        self.MAX_SPEED = 5.0
        self.MAX_REVERSE_SPEED = -2.0
        self.WALL_BOUNCE = 0.5
        self.WALL_SPEED_PENALTY = 0.5
        
        # Drift constants
        self.DRIFT_TURN_MOD = 1.5
        self.DRIFT_FRICTION = 0.99
        self.DRIFT_SLIP = 0.8
        self.BOOST_METER_GAIN = 1.5
        
        # Boost constants
        self.BOOST_ACCELERATION = 0.3
        self.MAX_BOOST_SPEED = 10.0
        self.BOOST_METER_DECAY = 1.0
        self.MAX_BOOST = 100

        # Game constants
        self.MAX_STEPS = 10000 # Generous step limit
        self.TIME_LIMIT_SECONDS = 60
        self.TIME_LIMIT_FRAMES = self.TIME_LIMIT_SECONDS * 30
        self.LAPS_TO_WIN = 3
        self.TRACK_WIDTH = 80
        self.NUM_COINS = 30
        
        # State variables are initialized in reset()
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = 0
        self.laps = 0
        self.win = False
        self.player_pos = np.array([0.0, 0.0])
        self.player_speed = 0.0
        self.player_angle = 0.0
        self.player_velocity = np.array([0.0, 0.0])
        self.is_drifting = False
        self.is_boosting = False
        self.boost_meter = 0.0
        self.next_checkpoint = 0
        self.track_points = []
        self.track_centerline = []
        self.track_poly = []
        self.coins = []
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def _catmull_rom_spline(self, P0, P1, P2, P3, num_points=20):
        """Generates points for a Catmull-Rom spline segment."""
        points = []
        for t in np.linspace(0, 1, num_points):
            t2 = t * t
            t3 = t2 * t
            
            c1 = -0.5 * t3 + t2 - 0.5 * t
            c2 = 1.5 * t3 - 2.5 * t2 + 1.0
            c3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            c4 = 0.5 * t3 - 0.5 * t2
            
            x = c1 * P0[0] + c2 * P1[0] + c3 * P2[0] + c4 * P3[0]
            y = c1 * P0[1] + c2 * P1[1] + c3 * P2[1] + c4 * P3[1]
            points.append(np.array([x, y]))
        return points

    def _generate_track(self):
        """Procedurally generates a closed-loop track."""
        num_control_points = 12
        center_x, center_y = self.SCREEN_WIDTH * 1.5, self.SCREEN_HEIGHT * 1.5
        radius_mean = min(center_x, center_y) * 0.8
        radius_variance = radius_mean * 0.4

        control_points = []
        for i in range(num_control_points):
            angle = 2 * math.pi * i / num_control_points
            radius = self.np_random.uniform(radius_mean - radius_variance, radius_mean + radius_variance)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            control_points.append(np.array([x, y]))
        
        self.track_points = control_points
        self.track_centerline = []
        
        # Create a closed loop by wrapping control points
        for i in range(num_control_points):
            p0 = control_points[(i - 1 + num_control_points) % num_control_points]
            p1 = control_points[i]
            p2 = control_points[(i + 1) % num_control_points]
            p3 = control_points[(i + 2) % num_control_points]
            self.track_centerline.extend(self._catmull_rom_spline(p0, p1, p2, p3))
        
        # Generate the visual track polygon
        self.track_poly = []
        outer_edge = []
        inner_edge = []
        for i in range(len(self.track_centerline)):
            p1 = self.track_centerline[i]
            p2 = self.track_centerline[(i + 1) % len(self.track_centerline)]
            
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) + math.pi / 2
            offset = np.array([math.cos(angle), math.sin(angle)]) * self.TRACK_WIDTH / 2
            
            outer_edge.append(p1 + offset)
            inner_edge.append(p1 - offset)
        
        self.track_poly = outer_edge + inner_edge[::-1]

    def _place_coins(self):
        self.coins = []
        if not self.track_centerline: return

        indices = self.np_random.choice(len(self.track_centerline), self.NUM_COINS, replace=False)
        for i in indices:
            if i == 0: continue # Avoid coin on start line
            point = self.track_centerline[i]
            offset_angle = self.np_random.uniform(0, 2 * math.pi)
            offset_dist = self.np_random.uniform(0, self.TRACK_WIDTH / 3)
            pos = point + np.array([math.cos(offset_angle), math.sin(offset_angle)]) * offset_dist
            self.coins.append({"pos": pos, "radius": 8, "collected_timer": 0})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.TIME_LIMIT_FRAMES
        self.laps = 0
        self.win = False

        self._generate_track()
        self._place_coins()

        self.player_pos = self.track_centerline[1].copy() # Start just after the line
        p_start = self.track_centerline[0]
        p_next = self.track_centerline[1]
        self.player_angle = math.atan2(p_next[1] - p_start[1], p_next[0] - p_start[0])
        self.player_speed = 0.0
        self.player_velocity = np.array([0.0, 0.0])
        
        self.is_drifting = False
        self.is_boosting = False
        self.boost_meter = 0.0
        self.next_checkpoint = 1

        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = -0.01 # Time penalty
        
        self._handle_input_and_physics(movement, space_held, shift_held)
        coin_collected_reward = self._handle_collisions_and_progress()
        reward += coin_collected_reward

        lap_completed_reward = self._update_lap_counter()
        reward += lap_completed_reward

        self._update_particles()
        
        self.steps += 1
        self.time_left -= 1
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 50
            else:
                reward -= 50
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input_and_physics(self, movement, space_held, shift_held):
        # Handle boosting
        self.is_boosting = shift_held and self.boost_meter > 0
        if self.is_boosting:
            self.boost_meter = max(0, self.boost_meter - self.BOOST_METER_DECAY)
            current_max_speed = self.MAX_BOOST_SPEED
            current_acceleration = self.BOOST_ACCELERATION
            # Add boost particles
            if self.np_random.random() < 0.5:
                self.particles.append(self._create_particle(self.player_pos, type="boost"))
        else:
            current_max_speed = self.MAX_SPEED
            current_acceleration = self.ACCELERATION

        # Handle acceleration/braking
        if movement == 1: # Up
            self.player_speed = min(current_max_speed, self.player_speed + current_acceleration)
        elif movement == 2: # Down
            self.player_speed = max(self.MAX_REVERSE_SPEED, self.player_speed - self.BRAKE_POWER)
        
        # Handle steering and drifting
        steer_input = 0
        if movement == 3: steer_input = -1 # Left
        if movement == 4: steer_input = 1  # Right
        
        is_turning = steer_input != 0 and abs(self.player_speed) > 0.5
        self.is_drifting = space_held and is_turning

        turn_rate = self.TURN_SPEED * (1 - self.player_speed / (current_max_speed * 2))
        
        if self.is_drifting:
            self.player_angle += steer_input * turn_rate * self.DRIFT_TURN_MOD
            self.boost_meter = min(self.MAX_BOOST, self.boost_meter + self.BOOST_METER_GAIN)
            # Add drift particles
            if self.np_random.random() < 0.7:
                 self.particles.append(self._create_particle(self.player_pos, type="drift"))
        elif is_turning:
            self.player_angle += steer_input * turn_rate

        # Update physics
        car_direction = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        
        if self.is_drifting:
            self.player_velocity = self.player_velocity * self.DRIFT_FRICTION
            # Blend velocity towards car direction
            self.player_velocity = self.player_velocity * self.DRIFT_SLIP + car_direction * self.player_speed * (1 - self.DRIFT_SLIP)
        else:
            self.player_velocity = car_direction * self.player_speed
            self.player_speed *= self.FRICTION

        self.player_pos += self.player_velocity

    def _handle_collisions_and_progress(self):
        # Wall collisions
        min_dist = float('inf')
        closest_point = None
        for p in self.track_centerline:
            dist = np.linalg.norm(self.player_pos - p)
            if dist < min_dist:
                min_dist = dist
                closest_point = p
        
        if min_dist > self.TRACK_WIDTH / 2:
            # Collision detected
            normal = self.player_pos - closest_point
            normal /= np.linalg.norm(normal)
            self.player_pos = closest_point + normal * (self.TRACK_WIDTH / 2)
            
            # Reflect velocity
            self.player_velocity = self.player_velocity - 2 * np.dot(self.player_velocity, normal) * normal * self.WALL_BOUNCE
            self.player_speed *= self.WALL_SPEED_PENALTY
            # sfx: car_hit_wall.wav

        # Coin collisions
        collected_reward = 0
        for coin in self.coins:
            if coin["collected_timer"] == 0:
                if np.linalg.norm(self.player_pos - coin["pos"]) < 10 + coin["radius"]:
                    coin["collected_timer"] = 1 # Start collection animation
                    self.score += 1
                    collected_reward += 0.1
                    # sfx: coin_collect.wav
                    for _ in range(10):
                        self.particles.append(self._create_particle(coin["pos"], type="coin"))
        
        return collected_reward

    def _update_lap_counter(self):
        checkpoint_radius = self.TRACK_WIDTH * 1.5
        num_checkpoints = len(self.track_points)
        
        target_checkpoint_pos = self.track_points[self.next_checkpoint]
        if np.linalg.norm(self.player_pos - target_checkpoint_pos) < checkpoint_radius:
            self.next_checkpoint = (self.next_checkpoint + 1) % num_checkpoints
            
            if self.next_checkpoint == 1: # Passed the finish line (checkpoint 0)
                self.laps += 1
                return 1.0 # Lap completion reward
        return 0.0

    def _check_termination(self):
        if self.laps >= self.LAPS_TO_WIN:
            self.win = True
            self.game_over = True
        elif self.time_left <= 0:
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _create_particle(self, pos, type):
        if type == "drift":
            return {
                "pos": pos.copy() + self.np_random.uniform(-5, 5, 2),
                "vel": -self.player_velocity * 0.2 + self.np_random.uniform(-0.5, 0.5, 2),
                "life": 15, "max_life": 15, "color": (255, 255, 255), "radius": 3
            }
        elif type == "coin":
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            return {
                "pos": pos.copy(), "vel": np.array([math.cos(angle), math.sin(angle)]) * speed,
                "life": 20, "max_life": 20, "color": self.COLOR_COIN, "radius": 4
            }
        elif type == "boost":
            return {
                "pos": pos.copy() - self.player_velocity * 0.5,
                "vel": -self.player_velocity * 0.1 + self.np_random.uniform(-1, 1, 2),
                "life": 25, "max_life": 25, "color": self.COLOR_BOOST_PARTICLE, "radius": 5
            }
        return None

    def _update_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] > 0:
                active_particles.append(p)
        self.particles = active_particles

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        camera_offset = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2]) - self.player_pos
        
        self._render_game(camera_offset)
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self, camera_offset):
        # Render track
        if self.track_poly:
            track_points_on_screen = [p + camera_offset for p in self.track_poly]
            pygame.gfxdraw.filled_polygon(self.screen, track_points_on_screen, self.COLOR_TRACK)
            pygame.gfxdraw.aapolygon(self.screen, track_points_on_screen, self.COLOR_WALL)

        # Render finish line
        if len(self.track_centerline) > 1:
            p1 = self.track_centerline[0]
            p2 = self.track_centerline[1]
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) + math.pi / 2
            offset = np.array([math.cos(angle), math.sin(angle)]) * self.TRACK_WIDTH / 2
            start = p1 - offset + camera_offset
            end = p1 + offset + camera_offset
            pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), 3)

        # Render coins
        for coin in self.coins:
            if coin["collected_timer"] == 0:
                pos = coin["pos"] + camera_offset
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), coin["radius"], self.COLOR_COIN)
                pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), coin["radius"], self.COLOR_COIN)

        # Render particles
        for p in self.particles:
            pos = p["pos"] + camera_offset
            alpha = int(255 * (p["life"] / p["max_life"]))
            color = (*p["color"], alpha)
            radius = int(p["radius"] * (p["life"] / p["max_life"]))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), radius, color)

        # Render player
        player_screen_pos = (self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        
        car_size = (20, 10)
        car_surf = pygame.Surface(car_size, pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self.COLOR_PLAYER, (0, 0, *car_size), border_radius=3)
        
        # Glow effect
        glow_surf = pygame.Surface((car_size[0] + 8, car_size[1] + 8), pygame.SRCALPHA)
        pygame.draw.rect(glow_surf, self.COLOR_PLAYER_GLOW, (0, 0, car_size[0]+8, car_size[1]+8), border_radius=5)
        
        rotated_glow = pygame.transform.rotate(glow_surf, -math.degrees(self.player_angle))
        rotated_car = pygame.transform.rotate(car_surf, -math.degrees(self.player_angle))
        
        self.screen.blit(rotated_glow, rotated_glow.get_rect(center=player_screen_pos))
        self.screen.blit(rotated_car, rotated_car.get_rect(center=player_screen_pos))

    def _render_ui(self):
        # Laps
        lap_text = self.font_medium.render(f"LAP: {min(self.laps, self.LAPS_TO_WIN)} / {self.LAPS_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lap_text, (10, 10))

        # Time
        time_sec = self.time_left // 30
        time_text = self.font_large.render(f"{time_sec:02d}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 5))

        # Coins
        coin_text = self.font_medium.render(f"COINS: {self.score}", True, self.COLOR_COIN)
        self.screen.blit(coin_text, (self.SCREEN_WIDTH // 2 - coin_text.get_width() // 2, self.SCREEN_HEIGHT - 40))

        # Speed
        speed_kmh = int(np.linalg.norm(self.player_velocity) * 20)
        speed_text = self.font_small.render(f"{speed_kmh} KM/H", True, self.COLOR_UI_TEXT)
        self.screen.blit(speed_text, (self.SCREEN_WIDTH - speed_text.get_width() - 10, self.SCREEN_HEIGHT - 30))
        
        # Boost Meter
        bar_width = 150
        bar_height = 15
        boost_fill = (self.boost_meter / self.MAX_BOOST) * bar_width
        boost_rect_bg = pygame.Rect(10, self.SCREEN_HEIGHT - bar_height - 10, bar_width, bar_height)
        boost_rect_fill = pygame.Rect(10, self.SCREEN_HEIGHT - bar_height - 10, boost_fill, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_WALL, boost_rect_bg, border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_BOOST_BAR, boost_rect_fill, border_radius=3)
        boost_text = self.font_small.render("BOOST", True, self.COLOR_UI_TEXT)
        self.screen.blit(boost_text, (15, self.SCREEN_HEIGHT - bar_height - 30))

        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU WIN!" if self.win else "TIME UP!"
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            self.screen.blit(end_text, (self.SCREEN_WIDTH//2 - end_text.get_width()//2, self.SCREEN_HEIGHT//2 - end_text.get_height()//2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "laps": self.laps,
            "time_left": self.time_left / 30,
            "boost": self.boost_meter
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part is for human testing and requires a display.
    # It will not run in a headless environment.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"]
    except KeyError:
        print("Running in interactive mode. Use arrow keys, space, and shift.")
        
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption(env.game_description)
        
        obs, info = env.reset()
        terminated = False
        
        while not terminated:
            keys = pygame.key.get_pressed()
            
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Draw the observation from the environment
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    obs, info = env.reset()

            env.clock.tick(30) # Limit to 30 FPS for manual play
            
        env.close()