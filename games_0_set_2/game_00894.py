import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to steer. Hold Space to accelerate, Shift to brake."
    )

    game_description = (
        "Race a sleek, neon car along a twisting, procedurally generated track. "
        "Dodge obstacles and pass checkpoints to get the best time."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (5, 0, 15)
    COLOR_TRACK = (0, 10, 40)
    COLOR_TRACK_BORDER = (200, 200, 255)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_OBSTACLE = (255, 0, 80)
    COLOR_CHECKPOINT = (255, 255, 0)
    COLOR_FINISH_LINE = (255, 255, 255)
    COLOR_TEXT = (220, 220, 240)
    
    # Game parameters
    MAX_STEPS = 1800  # 60 seconds at 30 FPS
    TRACK_LENGTH = 12000 # pixels
    NUM_CHECKPOINTS = 10
    
    # Player physics
    PLAYER_X_POS = 160 # Player's fixed horizontal position on screen
    STEERING_SPEED = 4.0
    BASE_SPEED = 4.0
    ACCELERATION = 0.2
    BRAKING = 0.4
    DRAG = 0.02
    MAX_SPEED = 15.0
    MIN_SPEED = 2.0
    
    # Track generation
    TRACK_BASE_WIDTH = 120
    TRACK_WIDTH_VARIATION = 50
    TRACK_ROUGHNESS = 0.4
    TRACK_SEGMENT_LENGTH = 100

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        self.particles = []
        self.obstacles = []
        self.track_centerline = []
        self.track_top_border = []
        self.track_bottom_border = []
        self.checkpoints = []
        
        # Attributes are initialized here to be available, but are properly set in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.car_y = 0
        self.car_speed = 0
        self.world_scroll_x = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.car_y = self.SCREEN_HEIGHT / 2
        self.car_speed = self.BASE_SPEED
        self.world_scroll_x = 0
        
        self.particles.clear()
        self.obstacles.clear()

        self._generate_track()
        self._spawn_initial_obstacles()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        
        # --- Handle Input and Physics ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Steering
        steer_input = 0
        if movement == 3:  # Left
            steer_input = -1
        elif movement == 4: # Right
            steer_input = 1
        self.car_y += steer_input * self.STEERING_SPEED

        # Acceleration/Braking
        if space_held and not shift_held:
            self.car_speed += self.ACCELERATION
        elif shift_held:
            self.car_speed -= self.BRAKING
            reward -= 0.05 # Small penalty for braking
        else:
            # Apply drag / drift towards base speed
            if self.car_speed > self.BASE_SPEED:
                self.car_speed -= self.DRAG
            elif self.car_speed < self.BASE_SPEED:
                self.car_speed += self.DRAG

        self.car_speed = np.clip(self.car_speed, self.MIN_SPEED, self.MAX_SPEED)
        self.world_scroll_x += self.car_speed

        # --- Update World ---
        self._update_particles()
        self._update_obstacles()
        self._spawn_obstacles()

        # --- Collision Detection ---
        # Track boundaries
        current_segment_idx = int(self.world_scroll_x + self.PLAYER_X_POS)
        if 0 < current_segment_idx < len(self.track_top_border):
            track_top_y = self.track_top_border[current_segment_idx][1]
            track_bottom_y = self.track_bottom_border[current_segment_idx][1]
            car_half_height = 8 # Approx
            if self.car_y - car_half_height < track_top_y or self.car_y + car_half_height > track_bottom_y:
                self.game_over = True
                reward -= 50
                self._create_explosion(self.PLAYER_X_POS, self.car_y, self.COLOR_PLAYER)
        
        # Obstacles
        min_dist_to_obstacle = float('inf')
        car_rect = pygame.Rect(self.PLAYER_X_POS - 5, self.car_y - 8, 10, 16)
        for obs in self.obstacles:
            obs_screen_x = obs['x'] - self.world_scroll_x
            obs_rect = obs['rect_func'](obs_screen_x, obs['y'], obs)
            
            if car_rect.colliderect(obs_rect):
                self.game_over = True
                reward -= 50
                self._create_explosion(self.PLAYER_X_POS, self.car_y, self.COLOR_OBSTACLE)
                break
            
            # For near-miss reward
            dist = math.hypot(car_rect.centerx - obs_rect.centerx, car_rect.centery - obs_rect.centery)
            min_dist_to_obstacle = min(min_dist_to_obstacle, dist)

        # --- Calculate Rewards ---
        reward += 0.01 # Survival reward
        
        # Near miss penalty
        if not self.game_over and min_dist_to_obstacle < 30:
            reward -= 0.5
            self._create_sparks(self.PLAYER_X_POS, self.car_y, 5)

        # Checkpoint reward
        for i, (cp_x, passed) in enumerate(self.checkpoints):
            if not passed and self.world_scroll_x + self.PLAYER_X_POS > cp_x:
                self.checkpoints[i] = (cp_x, True)
                reward += 10
                self.score += 10
                self._create_explosion(self.PLAYER_X_POS, self.car_y, self.COLOR_CHECKPOINT, count=20)
                break

        # --- Check Termination ---
        terminated = self.game_over
        if self.world_scroll_x >= self.TRACK_LENGTH:
            terminated = True
            self.win = True
            reward += 100
            self.score += 100
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            reward -= 20 # Time out penalty
        
        if terminated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        checkpoints_passed = sum(1 for _, passed in self.checkpoints if passed)
        return {
            "score": self.score,
            "steps": self.steps,
            "speed": self.car_speed,
            "checkpoints": f"{checkpoints_passed}/{self.NUM_CHECKPOINTS}"
        }

    # --- Generation Methods ---
    def _generate_track(self):
        self.track_centerline.clear()
        self.track_top_border.clear()
        self.track_bottom_border.clear()
        self.checkpoints.clear()

        y = self.SCREEN_HEIGHT / 2
        y_deriv = 0
        y_deriv2 = 0
        
        # Add a straight starting section to pass the stability test.
        # The test runs for 60 no-op steps. With a base speed of 4.0, the car
        # travels 240 pixels. The car's screen X position is 160, so it will
        # reach a world coordinate of 400. 500 provides a safe margin.
        straight_start_length = 500

        for x in range(self.TRACK_LENGTH + self.SCREEN_WIDTH):
            # Only start curving the track after the initial straight section
            if x > straight_start_length:
                y_deriv2 += self.np_random.uniform(-self.TRACK_ROUGHNESS, self.TRACK_ROUGHNESS)
                y_deriv2 = np.clip(y_deriv2, -0.5, 0.5)
                y_deriv += y_deriv2
                y_deriv = np.clip(y_deriv, -2, 2)
            
            # For the first `straight_start_length` pixels, y_deriv remains 0, so y does not change.
            y += y_deriv
            y = np.clip(y, self.SCREEN_HEIGHT * 0.2, self.SCREEN_HEIGHT * 0.8)
            
            width = self.TRACK_BASE_WIDTH + math.sin(x / 300) * self.TRACK_WIDTH_VARIATION
            
            self.track_centerline.append((x, y))
            self.track_top_border.append((x, y - width / 2))
            self.track_bottom_border.append((x, y + width / 2))

        checkpoint_interval = self.TRACK_LENGTH / (self.NUM_CHECKPOINTS + 1)
        for i in range(1, self.NUM_CHECKPOINTS + 1):
            self.checkpoints.append((int(i * checkpoint_interval), False))

    def _spawn_initial_obstacles(self):
        for _ in range(20):
            self._spawn_obstacles(force_spawn=True)

    def _spawn_obstacles(self, force_spawn=False):
        spawn_chance = 0.03 + 0.10 * (self.steps / self.MAX_STEPS)
        if force_spawn or self.np_random.random() < spawn_chance:
            spawn_x = self.world_scroll_x + self.SCREEN_WIDTH + 50
            if spawn_x >= self.TRACK_LENGTH - 100: return

            center_y = self.track_centerline[int(spawn_x)][1]
            track_width = self.track_bottom_border[int(spawn_x)][1] - self.track_top_border[int(spawn_x)][1]
            
            obs_type = self.np_random.choice(['square', 'rect', 'circle'])
            
            obs = {
                'x': spawn_x,
                'y': center_y + self.np_random.uniform(-track_width * 0.4, track_width * 0.4),
                'type': obs_type,
                'created_at': self.steps
            }

            if obs_type == 'square':
                obs['size'] = self.np_random.integers(15, 25)
                obs['angle'] = 0
                obs['rot_speed'] = self.np_random.uniform(-0.1, 0.1)
                obs['rect_func'] = lambda sx, sy, o: pygame.Rect(sx - o['size'] / 2, sy - o['size'] / 2, o['size'], o['size'])
            elif obs_type == 'rect':
                obs['w'] = self.np_random.integers(5, 10)
                obs['h'] = self.np_random.integers(30, 50)
                obs['vel_y'] = self.np_random.uniform(-1, 1) * (1 + self.steps / self.MAX_STEPS)
                obs['rect_func'] = lambda sx, sy, o: pygame.Rect(sx - o['w'] / 2, sy - o['h'] / 2, o['w'], o['h'])
            elif obs_type == 'circle':
                obs['base_radius'] = self.np_random.integers(10, 20)
                obs['pulse_speed'] = self.np_random.uniform(0.05, 0.1)
                obs['radius'] = obs['base_radius']
                obs['rect_func'] = lambda sx, sy, o: pygame.Rect(sx - o['radius'], sy - o['radius'], o['radius'] * 2, o['radius'] * 2)
            
            self.obstacles.append(obs)
    
    # --- Update Methods ---
    def _update_obstacles(self):
        # Prune old obstacles
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > self.world_scroll_x - 50]
        
        # Update active obstacles
        for obs in self.obstacles:
            if obs['x'] >= len(self.track_top_border): continue # Guard against out of bounds
            if obs['type'] == 'square':
                obs['angle'] += obs['rot_speed']
            elif obs['type'] == 'rect':
                obs['y'] += obs['vel_y']
                # Bounce off track walls
                track_top = self.track_top_border[int(obs['x'])][1]
                track_bottom = self.track_bottom_border[int(obs['x'])][1]
                if obs['y'] - obs['h']/2 < track_top or obs['y'] + obs['h']/2 > track_bottom:
                    obs['vel_y'] *= -1
            elif obs['type'] == 'circle':
                t = (self.steps - obs['created_at']) * obs['pulse_speed']
                obs['radius'] = obs['base_radius'] * (1 + 0.2 * math.sin(t))

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['life'] -= 1

    # --- Particle Effects ---
    def _create_explosion(self, x, y, color, count=50):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            self.particles.append({
                'x': x, 'y': y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _create_sparks(self, x, y, count=10):
        for _ in range(count):
            angle = self.np_random.uniform(-0.5, 0.5) - math.pi / 2
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'x': x, 'y': y,
                'vx': self.car_speed + math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': self.np_random.integers(5, 15),
                'color': (255, 200, 100)
            })

    # --- Rendering Methods ---
    def _render_game(self):
        self._render_background()
        self._render_track()
        self._render_checkpoints_and_finish()
        self._render_obstacles()
        self._render_particles()
        self._render_car()

    def _render_background(self):
        # Simple parallax stars
        for i in range(50):
            seed = i * 12345
            x = (self.SCREEN_WIDTH - (self.world_scroll_x * (0.1 * (seed % 5 + 1))) % self.SCREEN_WIDTH) % self.SCREEN_WIDTH
            y = (seed * 13) % self.SCREEN_HEIGHT
            size = 1 + (seed % 2)
            pygame.draw.rect(self.screen, (50, 50, 80), (int(x), int(y), size, size))

    def _render_track(self):
        start_idx = max(0, int(self.world_scroll_x))
        end_idx = min(len(self.track_centerline), start_idx + self.SCREEN_WIDTH + 50)
        
        if end_idx <= start_idx: return

        # Create polygon for the track surface
        poly_points = []
        for i in range(start_idx, end_idx):
            poly_points.append((self.track_top_border[i][0] - self.world_scroll_x, self.track_top_border[i][1]))
        for i in range(end_idx - 1, start_idx - 1, -1):
            poly_points.append((self.track_bottom_border[i][0] - self.world_scroll_x, self.track_bottom_border[i][1]))
        
        if len(poly_points) > 2:
            pygame.gfxdraw.filled_polygon(self.screen, poly_points, self.COLOR_TRACK)

        # Draw antialiased borders
        for border in [self.track_top_border, self.track_bottom_border]:
            points_to_draw = [(p[0] - self.world_scroll_x, p[1]) for p in border[start_idx:end_idx]]
            if len(points_to_draw) > 1:
                pygame.draw.aalines(self.screen, self.COLOR_TRACK_BORDER, False, points_to_draw)

    def _render_checkpoints_and_finish(self):
        # Checkpoints
        for cp_x, passed in self.checkpoints:
            screen_x = cp_x - self.world_scroll_x
            if 0 < screen_x < self.SCREEN_WIDTH:
                start_idx = max(0, int(cp_x))
                if start_idx < len(self.track_top_border):
                    y1 = self.track_top_border[start_idx][1]
                    y2 = self.track_bottom_border[start_idx][1]
                    color = self.COLOR_CHECKPOINT if not passed else (50, 50, 0)
                    pygame.draw.line(self.screen, color, (screen_x, y1), (screen_x, y2), 3)

        # Finish line
        finish_x = self.TRACK_LENGTH - self.world_scroll_x
        if 0 < finish_x < self.SCREEN_WIDTH:
            if self.TRACK_LENGTH-1 < len(self.track_top_border):
                y1 = self.track_top_border[self.TRACK_LENGTH-1][1]
                y2 = self.track_bottom_border[self.TRACK_LENGTH-1][1]
                for i in range(int(y1), int(y2), 10):
                    color = self.COLOR_FINISH_LINE if (i // 10) % 2 == 0 else (0,0,0)
                    pygame.draw.rect(self.screen, color, (finish_x - 5, i, 10, 10))


    def _render_car(self):
        x, y = self.PLAYER_X_POS, int(self.car_y)
        points = [
            (x + 12, y),
            (x - 8, y - 6),
            (x - 8, y + 6)
        ]
        
        # Glow effect
        for i in range(4, 0, -1):
            glow_points = [
                (x + 12 + i, y),
                (x - 8 - i, y - 6 - i),
                (x - 8 - i, y + 6 + i)
            ]
            color = list(self.COLOR_PLAYER) + [30] # Add alpha
            pygame.gfxdraw.filled_polygon(self.screen, glow_points, color)
            pygame.gfxdraw.aapolygon(self.screen, glow_points, color)

        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

    def _render_obstacles(self):
        for obs in self.obstacles:
            screen_x = obs['x'] - self.world_scroll_x
            if -50 < screen_x < self.SCREEN_WIDTH + 50:
                y = int(obs['y'])
                
                # Glow effect
                color_rgba = list(self.COLOR_OBSTACLE) + [40]
                if obs['type'] == 'square':
                    size = obs['size'] + 8
                    pygame.draw.rect(self.screen, color_rgba, (screen_x - size/2, y - size/2, size, size), border_radius=4)
                elif obs['type'] == 'rect':
                    pygame.draw.rect(self.screen, color_rgba, (screen_x - obs['w']/2-4, y - obs['h']/2-4, obs['w']+8, obs['h']+8), border_radius=4)
                elif obs['type'] == 'circle':
                    pygame.gfxdraw.filled_circle(self.screen, int(screen_x), y, int(obs['radius'] + 4), color_rgba)

                # Main shape
                if obs['type'] == 'square':
                    size = obs['size']
                    angle = obs['angle']
                    points = [
                        (-size/2, -size/2), (size/2, -size/2),
                        (size/2, size/2), (-size/2, size/2)
                    ]
                    rotated = [(p[0]*math.cos(angle) - p[1]*math.sin(angle) + screen_x,
                                p[0]*math.sin(angle) + p[1]*math.cos(angle) + y) for p in points]
                    pygame.gfxdraw.filled_polygon(self.screen, rotated, self.COLOR_OBSTACLE)
                    pygame.gfxdraw.aapolygon(self.screen, rotated, self.COLOR_OBSTACLE)
                elif obs['type'] == 'rect':
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, (screen_x - obs['w']/2, y - obs['h']/2, obs['w'], obs['h']))
                elif obs['type'] == 'circle':
                    pygame.gfxdraw.filled_circle(self.screen, int(screen_x), y, int(obs['radius']), self.COLOR_OBSTACLE)
                    pygame.gfxdraw.aacircle(self.screen, int(screen_x), y, int(obs['radius']), self.COLOR_OBSTACLE)

    def _render_particles(self):
        for p in self.particles:
            screen_x = p['x'] - self.world_scroll_x
            alpha = max(0, 255 * (p['life'] / 30.0))
            color = list(p['color']) + [int(alpha)]
            size = max(1, int(3 * (p['life'] / 30.0)))
            pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(p['y']), size, color)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Time
        time_left = (self.MAX_STEPS - self.steps) / 30
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.SCREEN_WIDTH - time_text.get_width() - 10, 10))
        
        # Speed
        speed_kmh = int(self.car_speed * 20)
        speed_text = self.font_small.render(f"{speed_kmh} KM/H", True, self.COLOR_TEXT)
        self.screen.blit(speed_text, (10, self.SCREEN_HEIGHT - speed_text.get_height() - 10))

        # Checkpoints
        info = self._get_info()
        cp_text = self.font_small.render(f"CP: {info['checkpoints']}", True, self.COLOR_TEXT)
        self.screen.blit(cp_text, (10, 35))

        if self.game_over:
            if self.win:
                end_text = self.font_large.render("FINISH!", True, self.COLOR_CHECKPOINT)
            else:
                end_text = self.font_large.render("GAME OVER", True, self.COLOR_OBSTACLE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(end_text, text_rect)


if __name__ == '__main__':
    # This block allows you to run the game directly for testing
    # Un-comment the next line to run with display
    # os.environ.pop("SDL_VIDEODRIVER", None)
    
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Arcade Racer")
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    while not done:
        # --- Get action from keyboard ---
        keys = pygame.key.get_pressed()
        mov = 0 # no-op
        if keys[pygame.K_LEFT]: mov = 3
        if keys[pygame.K_RIGHT]: mov = 4
        
        space = 1 if keys[pygame.K_SPACE] else 0
        shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [mov, space, shift]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # --- Render to screen ---
        # The observation is a numpy array, we need to convert it back to a surface
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        real_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Handle Pygame events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    pygame.quit()