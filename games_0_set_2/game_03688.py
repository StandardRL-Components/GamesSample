import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Use arrow keys to accelerate your car. Hold Shift to brake, and press Space for a speed boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade racer. Race against the clock on a procedural track, collecting coins and dodging obstacles to reach the finish line."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    # --- Constants ---
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_TRACK = (60, 60, 70)
    COLOR_TRACK_BORDER = (180, 180, 180)
    COLOR_PLAYER = (50, 255, 100)
    COLOR_PLAYER_BOOST = (100, 255, 255)
    COLOR_OBSTACLE = (255, 50, 50)
    COLOR_COIN = (255, 223, 0)
    COLOR_FINISH_1 = (255, 255, 255)
    COLOR_FINISH_2 = (30, 30, 30)
    COLOR_TEXT = (240, 240, 240)
    COLOR_SPARK = (255, 180, 50)

    # Screen dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game parameters
    MAX_STEPS = 900  # 30 seconds at 30 FPS
    START_TIME = 30.0
    MAX_CRASHES = 5
    
    # Player physics
    PLAYER_SIZE = 12
    ACCELERATION = 0.4
    BOOST_FORCE = 1.5
    FRICTION = 0.96
    BRAKE_FRICTION = 0.85
    MAX_SPEED = 8.0
    
    # Track & Entity parameters
    TRACK_WIDTH = 80
    COIN_RADIUS = 8
    OBSTACLE_RADIUS = 10
    BASE_OBSTACLE_COUNT = 5
    FINISH_LINE_WIDTH = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        if "SDL_VIDEODRIVER" not in os.environ:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)
        
        # Etc...        
        self.player_pos = None
        self.player_vel = None
        self.coins = None
        self.obstacles = None
        self.particles = None
        self.track_points = None
        self.finish_line_poly = None
        self.last_obstacle_spawn_tier = 0
        self.boost_cooldown = 0
        self.game_won = False
        self.coins_collected = 0
        self.crashes = 0
        self.time_remaining = 0
        
        # Initialize state variables
        self.np_random = None # Will be seeded in reset
        self.steps = 0
        self.score = 0.0
        self.game_over = False
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.game_won = False
        self.time_remaining = self.START_TIME
        self.crashes = 0
        self.coins_collected = 0
        self.last_obstacle_spawn_tier = 0
        self.boost_cooldown = 0
        self.particles = []

        self._generate_track()
        self.player_pos = np.array(self.track_points[0], dtype=float) + [0, self.np_random.uniform(-self.TRACK_WIDTH/4, self.TRACK_WIDTH/4)]
        self.player_vel = np.array([2.0, 0.0], dtype=float)

        self._spawn_entities()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.01  # Cost of living
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            shift_held = action[2] == 1
            
            # Update game logic
            self._handle_input(movement, space_held, shift_held)
            self._update_player()
            reward += self._handle_collisions()

            self.time_remaining -= 1.0 / 30.0
            if self.boost_cooldown > 0:
                self.boost_cooldown -= 1

            time_elapsed = self.START_TIME - self.time_remaining
            current_tier = int(time_elapsed / 5)
            if current_tier > self.last_obstacle_spawn_tier:
                self.last_obstacle_spawn_tier = current_tier
                num_new_obstacles = int(self.BASE_OBSTACLE_COUNT * 0.05 * current_tier)
                self._spawn_obstacles(num_new_obstacles, start_index=len(self.track_points)//3)
        
        self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.game_won:
                reward += 15.0 # Win bonus
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "coins": self.coins_collected,
            "crashes": self.crashes,
            "time": self.time_remaining,
        }

    # --- Private Helper Methods ---

    def _generate_track(self):
        self.track_points = []
        x_start, y_start = 50, self.SCREEN_HEIGHT / 2
        self.track_points.append((x_start, y_start))
        
        num_segments = 8
        current_pos = np.array([x_start, y_start])
        
        for i in range(num_segments):
            segment_length = (self.SCREEN_WIDTH - 150) / num_segments
            x_offset = segment_length
            y_offset = self.np_random.uniform(-80, 80)
            
            next_pos = current_pos + [x_offset, y_offset]
            next_pos[1] = np.clip(next_pos[1], self.TRACK_WIDTH, self.SCREEN_HEIGHT - self.TRACK_WIDTH)
            
            self.track_points.append(tuple(next_pos))
            current_pos = next_pos
        
        p1 = self.track_points[-1]
        p0 = self.track_points[-2]
        angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        
        x, y = p1
        w = self.FINISH_LINE_WIDTH / 2
        
        self.finish_line_poly = [
            (x - w * math.sin(angle), y + w * math.cos(angle)),
            (x + w * math.sin(angle), y - w * math.cos(angle)),
            (x + w * math.sin(angle) + 10 * math.cos(angle), y - w * math.cos(angle) + 10 * math.sin(angle)),
            (x - w * math.sin(angle) + 10 * math.cos(angle), y + w * math.cos(angle) + 10 * math.sin(angle)),
        ]

    def _spawn_entities(self):
        self.coins = []
        self.obstacles = []
        
        for i in range(1, len(self.track_points) - 1):
            for _ in range(3):
                p1 = np.array(self.track_points[i])
                p2 = np.array(self.track_points[i+1])
                t = self.np_random.random()
                center_pos = p1 + t * (p2 - p1)
                angle = self.np_random.uniform(0, 2 * math.pi)
                radius = self.np_random.uniform(0, self.TRACK_WIDTH / 2.5)
                coin_pos = center_pos + [math.cos(angle) * radius, math.sin(angle) * radius]
                self.coins.append(list(coin_pos))

        self._spawn_obstacles(self.BASE_OBSTACLE_COUNT, start_index=2)

    def _spawn_obstacles(self, count, start_index=0):
        for _ in range(count):
            if len(self.track_points) <= start_index + 1: continue
            segment_idx = self.np_random.integers(start_index, len(self.track_points) - 1)
            p1, p2 = np.array(self.track_points[segment_idx]), np.array(self.track_points[segment_idx+1])
            t = self.np_random.random()
            center_pos = p1 + t * (p2 - p1)
            angle, radius = self.np_random.uniform(0, 2 * math.pi), self.np_random.uniform(0, self.TRACK_WIDTH / 2.2)
            obs_pos = center_pos + [math.cos(angle) * radius, math.sin(angle) * radius]
            
            too_close = any(np.linalg.norm(np.array(obs_pos) - np.array(o)) < self.OBSTACLE_RADIUS * 3 for o in self.obstacles)
            if not too_close:
                self.obstacles.append(list(obs_pos))

    def _handle_input(self, movement, space_held, shift_held):
        accel_vec = np.array([0.0, 0.0])
        if movement == 1: accel_vec[1] -= self.ACCELERATION
        elif movement == 2: accel_vec[1] += self.ACCELERATION
        elif movement == 3: accel_vec[0] -= self.ACCELERATION
        elif movement == 4: accel_vec[0] += self.ACCELERATION
        self.player_vel += accel_vec

        if shift_held: self.player_vel *= self.BRAKE_FRICTION
        
        if space_held and self.boost_cooldown == 0:
            # sfx: boost.wav
            vel_norm = np.linalg.norm(self.player_vel)
            boost_dir = self.player_vel / vel_norm if vel_norm > 0.5 else np.array([1.0, 0.0])
            self.player_vel += boost_dir * self.BOOST_FORCE
            self.boost_cooldown = 15
            self._create_particles(self.player_pos, 10, self.COLOR_PLAYER_BOOST, 3.0)

    def _update_player(self):
        self.player_vel *= self.FRICTION
        speed = np.linalg.norm(self.player_vel)
        if speed > self.MAX_SPEED:
            self.player_vel = self.player_vel / speed * self.MAX_SPEED
        self.player_pos += self.player_vel

    def _handle_collisions(self):
        reward = 0.0
        
        for coin_pos in self.coins[:]:
            if np.linalg.norm(self.player_pos - coin_pos) < self.PLAYER_SIZE / 2 + self.COIN_RADIUS:
                self.coins.remove(coin_pos)
                self.coins_collected += 1
                reward += 0.1
                # sfx: coin.wav
                self._create_particles(coin_pos, 5, self.COLOR_COIN, 1.5)
        
        for obs_pos in self.obstacles:
            if np.linalg.norm(self.player_pos - obs_pos) < self.PLAYER_SIZE / 2 + self.OBSTACLE_RADIUS:
                self.crashes += 1
                reward -= 1.0
                self.player_vel *= -0.5
                # sfx: crash.wav
                self._create_particles(self.player_pos, 20, self.COLOR_SPARK, 4.0)
                direction = self.player_pos - obs_pos
                dist = np.linalg.norm(direction)
                if dist > 0: self.player_pos += (direction / dist) * (self.PLAYER_SIZE/2 + self.OBSTACLE_RADIUS - dist + 1)
                break

        if not self._is_on_track(self.player_pos):
            self.crashes += 1
            reward -= 1.0
            self.player_vel *= -0.7
            # sfx: scrape.wav
            self._create_particles(self.player_pos, 10, self.COLOR_TRACK_BORDER, 2.0)
            min_dist, closest_point = float('inf'), None
            for i in range(len(self.track_points) - 1):
                p, d = self._closest_point_on_segment(self.player_pos, self.track_points[i], self.track_points[i+1])
                if d < min_dist: min_dist, closest_point = d, p
            if closest_point is not None:
                direction_to_track = closest_point - self.player_pos
                self.player_pos += direction_to_track * (1 - (self.TRACK_WIDTH/2 - 1) / min_dist)

        if self._point_in_poly(self.player_pos, self.finish_line_poly):
            self.game_won = True
            # sfx: win.wav

        return reward

    def _is_on_track(self, point):
        min_dist_sq = float('inf')
        for i in range(len(self.track_points) - 1):
            p1, p2 = np.array(self.track_points[i]), np.array(self.track_points[i+1])
            l2 = np.sum((p1 - p2)**2)
            if l2 == 0:
                min_dist_sq = min(min_dist_sq, np.sum((point - p1)**2))
                continue
            t = max(0, min(1, np.dot(point - p1, p2 - p1) / l2))
            projection = p1 + t * (p2 - p1)
            min_dist_sq = min(min_dist_sq, np.sum((point - projection)**2))
        return min_dist_sq < (self.TRACK_WIDTH / 2)**2

    def _closest_point_on_segment(self, p, a, b):
        p, a, b = np.array(p), np.array(a), np.array(b)
        ab_sq = np.dot(b - a, b - a)
        if ab_sq == 0: return a, np.linalg.norm(p - a)
        t = np.clip(np.dot(p - a, b - a) / ab_sq, 0, 1)
        closest = a + t * (b - a)
        return closest, np.linalg.norm(p - closest)

    def _point_in_poly(self, point, poly):
        x, y = point; n = len(poly); inside = False
        p1x, p1y = poly[0]
        for i in range(n + 1):
            p2x, p2y = poly[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y: xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters: inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _check_termination(self):
        return self.game_won or self.crashes >= self.MAX_CRASHES or self.time_remaining <= 0 or self.steps >= self.MAX_STEPS

    def _create_particles(self, pos, count, color, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': self.np_random.integers(10, 20), 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]; p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95; p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0: self.particles.remove(p)

    def _render_game(self):
        # Draw the wider border first
        pygame.draw.lines(self.screen, self.COLOR_TRACK_BORDER, False, self.track_points, self.TRACK_WIDTH + 2)
        # Then draw the narrower track on top
        pygame.draw.lines(self.screen, self.COLOR_TRACK, False, self.track_points, self.TRACK_WIDTH)
        
        pygame.draw.polygon(self.screen, self.COLOR_FINISH_1, self.finish_line_poly)
        p0, p1, p2, p3 = [np.array(p) for p in self.finish_line_poly]
        for i in range(5):
            if i % 2 == 1:
                t = i / 4.0; start = p0 + (p1-p0) * t; end = p3 + (p2-p3) * t
                pygame.draw.line(self.screen, self.COLOR_FINISH_2, start, end, int(self.FINISH_LINE_WIDTH/5))

        for pos in self.coins:
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.COIN_RADIUS, self.COLOR_COIN)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.COIN_RADIUS, self.COLOR_COIN)

        for pos in self.obstacles:
            pygame.gfxdraw.filled_circle(self.screen, int(pos[0]), int(pos[1]), self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE)
            pygame.gfxdraw.aacircle(self.screen, int(pos[0]), int(pos[1]), self.OBSTACLE_RADIUS, self.COLOR_OBSTACLE)

        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20.0))))
            size = max(1, int(p['life'] / 4))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, (*p['color'], alpha), (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)))
            
        player_color = self.COLOR_PLAYER_BOOST if self.boost_cooldown > 10 else self.COLOR_PLAYER
        px, py = int(self.player_pos[0]), int(self.player_pos[1]); s = self.PLAYER_SIZE // 2
        car_rect = pygame.Rect(px - s, py - s, s*2, s*2)
        pygame.draw.rect(self.screen, player_color, car_rect, border_radius=2)
        pygame.draw.rect(self.screen, (255,255,255), car_rect, 1, border_radius=2)

    def _render_ui(self):
        time_surf = self.font_small.render(f"TIME: {max(0, self.time_remaining):.2f}", True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))
        coin_surf = self.font_small.render(f"COINS: {self.coins_collected}", True, self.COLOR_TEXT)
        self.screen.blit(coin_surf, (self.SCREEN_WIDTH - coin_surf.get_width() - 10, 10))
        crash_surf = self.font_small.render(f"CRASHES: {self.crashes}/{self.MAX_CRASHES}", True, self.COLOR_TEXT)
        self.screen.blit(crash_surf, (10, self.SCREEN_HEIGHT - crash_surf.get_height() - 10))

        if self._check_termination():
            msg, color = ("FINISH!", self.COLOR_PLAYER) if self.game_won else \
                         ("WRECKED!", self.COLOR_OBSTACLE) if self.crashes >= self.MAX_CRASHES else \
                         ("TIME UP!", self.COLOR_TEXT)
            msg_surf = self.font_large.render(msg, True, color)
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)))

    def close(self):
        pygame.font.quit()
        pygame.quit()
        

# Example of how to run the environment
if __name__ == "__main__":
    # Set to "dummy" to run headless, or remove/comment out to run with a display.
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 
    
    # --- Headless Test ---
    is_headless = os.getenv("SDL_VIDEODRIVER") == "dummy"
    if is_headless:
        print("Running headless test with random actions...")
        env = GameEnv()
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0
        frame_count = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frame_count += 1
            done = terminated or truncated

        print(f"Test finished in {frame_count} steps.")
        print(f"Final Info: {info}")
        print(f"Total Reward: {total_reward:.2f}")
        env.close()

    # --- Manual Play Test (requires display) ---
    else:
        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset(seed=42)
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption(env.game_description)
        clock = pygame.time.Clock()
        done = False

        print(env.user_guide)

        while not done:
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space, shift]
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting environment.")
                    obs, info = env.reset()

            clock.tick(30)
            
        print("Game Over. Press R to play again or close the window.")
        # Keep window open to see final score
        running = True
        while running:
            event = pygame.event.wait()
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # A simple loop to restart the game
                obs, info = env.reset()
                done = False
                while not done:
                    keys = pygame.key.get_pressed()
                    movement = 0
                    if keys[pygame.K_UP]: movement = 1
                    elif keys[pygame.K_DOWN]: movement = 2
                    elif keys[pygame.K_LEFT]: movement = 3
                    elif keys[pygame.K_RIGHT]: movement = 4
                    space = 1 if keys[pygame.K_SPACE] else 0
                    shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
                    action = [movement, space, shift]
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    for e in pygame.event.get():
                        if e.type == pygame.QUIT: 
                            done = True
                            running = False
                    clock.tick(30)
        
        env.close()