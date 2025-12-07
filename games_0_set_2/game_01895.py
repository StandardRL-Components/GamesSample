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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑/↓ to steer your ship. Hold SPACE to boost."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Race your spaceship to the finish line, dodging asteroids in a stunning retro-futuristic starfield. Manage your boost to get the best time, but be careful not to crash!"
    )

    # Frames auto-advance at a fixed rate for smooth gameplay.
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.RACE_DISTANCE = 8000
        self.MAX_STEPS = self.FPS * 60  # 60-second time limit
        self.MAX_COLLISIONS = 3

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Colors
        self.COLOR_BG = (15, 15, 30)
        self.COLOR_PLAYER = (50, 150, 255)
        self.COLOR_PLAYER_GLOW = (150, 200, 255)
        self.COLOR_ASTEROID = (100, 100, 110)
        self.COLOR_ASTEROID_OUTLINE = (70, 70, 80)
        self.COLOR_BOOST = (255, 150, 0)
        self.COLOR_BOOST_GLOW = (255, 200, 100)
        self.COLOR_TRAIL = (200, 220, 255)
        self.COLOR_FINISH_LINE = (0, 255, 100)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_UI_BAR = (200, 200, 200)
        self.COLOR_UI_BAR_FILL = (100, 200, 255)
        self.COLOR_UI_BOOST_FILL = (255, 150, 0)

        # Player physics
        self.PLAYER_ACCEL = 1.0
        self.PLAYER_DRAG = 0.92
        self.BASE_SPEED = 5.0
        self.BOOST_SPEED = 12.0
        self.MAX_BOOST_FUEL = 100.0
        self.BOOST_DRAIN = 1.5
        self.BOOST_RECHARGE = 0.4

        # Asteroid properties
        self.ASTEROID_COUNT = 15
        self.ASTEROID_BASE_SPEED = 1.0
        self.DIFFICULTY_INTERVAL = 100

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_vel_y = None
        self.distance_covered = None
        self.boost_fuel = None
        self.collisions = None
        self.asteroids = None
        self.asteroid_speed_multiplier = None
        self.particles = None
        self.stars = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = pygame.math.Vector2(self.WIDTH * 0.15, self.HEIGHT / 2)
        self.player_vel_y = 0.0
        self.distance_covered = 0.0
        self.boost_fuel = self.MAX_BOOST_FUEL
        self.collisions = 0
        self.asteroid_speed_multiplier = 1.0

        self.asteroids = []
        for _ in range(self.ASTEROID_COUNT):
            self._spawn_asteroid(initial=True)

        self.particles = []
        self._create_starfield()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # 1. Handle Input & Update Player
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # Apply vertical acceleration
        if movement == 1:  # Up
            self.player_vel_y -= self.PLAYER_ACCEL
        elif movement == 2:  # Down
            self.player_vel_y += self.PLAYER_ACCEL
        
        # Apply drag and update position
        self.player_vel_y *= self.PLAYER_DRAG
        self.player_pos.y += self.player_vel_y
        self.player_pos.y = np.clip(self.player_pos.y, 0, self.HEIGHT)

        # Handle boost and horizontal speed
        is_boosting = space_held and self.boost_fuel > 0
        if is_boosting:
            current_speed = self.BOOST_SPEED
            self.boost_fuel = max(0, self.boost_fuel - self.BOOST_DRAIN)
        else:
            current_speed = self.BASE_SPEED
            self.boost_fuel = min(self.MAX_BOOST_FUEL, self.boost_fuel + self.BOOST_RECHARGE)
        
        prev_distance = self.distance_covered
        self.distance_covered += current_speed
        
        # Reward for distance covered
        reward += (self.distance_covered - prev_distance) / 10.0

        # 2. Update Game World (unless game is already over)
        if not self.game_over:
            self._update_stars(current_speed)
            self._update_asteroids(current_speed)
            self._spawn_particles(is_boosting)

            # 3. Check for collisions
            collided = self._check_collisions()
            if collided:
                self.collisions += 1
                reward -= 5.0
                self._create_explosion(self.player_pos)

            # 4. Difficulty Scaling
            if self.steps > 0 and self.steps % self.DIFFICULTY_INTERVAL == 0:
                self.asteroid_speed_multiplier += 0.05
        
        self._update_particles() # Particles should update even after game over

        # 5. Check Termination and Truncation Conditions
        self.steps += 1
        terminated = False
        truncated = False

        if self.distance_covered >= self.RACE_DISTANCE:
            self.game_won = True
            terminated = True
            time_bonus = 50.0 * (self.MAX_STEPS - self.steps) / self.MAX_STEPS
            reward += max(0, time_bonus)
        elif self.collisions >= self.MAX_COLLISIONS:
            terminated = True
        
        if not terminated and self.steps >= self.MAX_STEPS:
            truncated = True

        if terminated or truncated:
            self.game_over = True

        self.score += reward
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "distance_covered": self.distance_covered,
            "collisions": self.collisions,
        }

    # --- Helper and Rendering Methods ---

    def _create_starfield(self):
        self.stars = []
        for i in range(200):
            # z determines speed and size: 1 (slow, small), 2 (medium), 3 (fast, large)
            z = self.np_random.uniform(1, 4)
            self.stars.append({
                'pos': pygame.math.Vector2(self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)),
                'z': z
            })

    def _spawn_asteroid(self, initial=False):
        size = self.np_random.uniform(15, 40)
        if initial:
            # Spawn initial asteroids in the right half of the screen to avoid immediate collisions.
            x_pos = self.np_random.uniform(self.WIDTH * 0.5, self.WIDTH)
        else:
            # Spawn new asteroids off-screen to the right.
            x_pos = self.WIDTH + size
        
        asteroid = {
            'pos': pygame.math.Vector2(x_pos, self.np_random.uniform(0, self.HEIGHT)),
            'size': size,
            'angle': self.np_random.uniform(0, 360),
            'rot_speed': self.np_random.uniform(-1, 1),
            'speed': self.np_random.uniform(0.5, 1.5) * self.ASTEROID_BASE_SPEED,
            'shape': self._create_asteroid_shape(size)
        }
        self.asteroids.append(asteroid)

    def _create_asteroid_shape(self, size):
        points = []
        num_vertices = self.np_random.integers(7, 12)
        for i in range(num_vertices):
            angle = i * (2 * math.pi / num_vertices)
            radius = self.np_random.uniform(size * 0.7, size)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append((x, y))
        return points

    def _update_stars(self, player_speed):
        for star in self.stars:
            star['pos'].x -= player_speed / (5 - star['z'])
            if star['pos'].x < 0:
                star['pos'].x = self.WIDTH
                star['pos'].y = self.np_random.uniform(0, self.HEIGHT)
    
    def _update_asteroids(self, player_speed):
        for asteroid in self.asteroids:
            asteroid['pos'].x -= (asteroid['speed'] * self.asteroid_speed_multiplier)
            asteroid['angle'] += asteroid['rot_speed']
            if asteroid['pos'].x < -asteroid['size']:
                self.asteroids.remove(asteroid)
                self._spawn_asteroid()

    def _check_collisions(self):
        player_rect = pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 8, 30, 16)
        for asteroid in self.asteroids:
            dist = self.player_pos.distance_to(asteroid['pos'])
            if dist < asteroid['size'] + 10: # Simple circular collision check
                return True
        return False

    def _spawn_particles(self, is_boosting):
        # Trail particles
        if self.steps % 2 == 0:
            p_pos = self.player_pos - pygame.math.Vector2(15, 0)
            p_vel = pygame.math.Vector2(self.np_random.uniform(-1, 0), self.np_random.uniform(-0.5, 0.5))
            if is_boosting:
                p_life = self.np_random.integers(20, 30)
                p_size = self.np_random.uniform(4, 7)
                p_color = self.COLOR_BOOST
            else:
                p_life = self.np_random.integers(10, 20)
                p_size = self.np_random.uniform(2, 4)
                p_color = self.COLOR_TRAIL
            self.particles.append({'pos': p_pos, 'vel': p_vel, 'life': p_life, 'size': p_size, 'color': p_color})

    def _create_explosion(self, position):
        colors = [self.COLOR_BOOST, self.COLOR_BOOST_GLOW, (255, 255, 255)]
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            vel = pygame.math.Vector2(math.cos(angle), math.sin(angle)) * speed
            life = self.np_random.integers(20, 40)
            size = self.np_random.uniform(2, 5)
            color_idx = self.np_random.integers(0, len(colors))
            color = colors[color_idx]
            self.particles.append({'pos': position.copy(), 'vel': vel, 'life': life, 'size': size, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _render_game(self):
        # Stars
        for star in self.stars:
            size = star['z'] * 0.5
            color_val = int(50 + (star['z'] - 1) * 50)
            color = (color_val, color_val, color_val + 50)
            pygame.draw.circle(self.screen, color, (int(star['pos'].x), int(star['pos'].y)), size)

        # Particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / 40))
            p_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(p_surf, (*p['color'], alpha), (p['size'], p['size']), p['size'])
            self.screen.blit(p_surf, (int(p['pos'].x - p['size']), int(p['pos'].y - p['size'])), special_flags=pygame.BLEND_RGBA_ADD)

        # Asteroids
        for asteroid in self.asteroids:
            self._draw_rotated_polygon(asteroid['pos'], asteroid['shape'], asteroid['angle'], self.COLOR_ASTEROID, self.COLOR_ASTEROID_OUTLINE)
        
        # Player Ship
        if not self.game_over or self.game_won:
            self._draw_player_ship()
        
        # Finish Line
        dist_to_finish = self.RACE_DISTANCE - self.distance_covered
        if dist_to_finish < self.WIDTH:
            finish_x = self.WIDTH - dist_to_finish
            if self.steps % 10 < 5: # Flashing effect
                pygame.draw.line(self.screen, self.COLOR_FINISH_LINE, (finish_x, 0), (finish_x, self.HEIGHT), 5)

    def _draw_player_ship(self):
        # Points for a triangular ship shape
        points = [(-15, -10), (15, 0), (-15, 10)]
        # Add a slight tilt based on vertical velocity
        angle = self.player_vel_y * 2
        
        # Glow effect
        glow_points = self._rotate_points(points, angle)
        glow_points = [(p[0] + self.player_pos.x, p[1] + self.player_pos.y) for p in glow_points]
        pygame.gfxdraw.aapolygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_polygon(self.screen, glow_points, self.COLOR_PLAYER_GLOW)
        
        # Main body
        body_points = self._rotate_points([(p[0]*0.9, p[1]*0.9) for p in points], angle)
        body_points = [(p[0] + self.player_pos.x, p[1] + self.player_pos.y) for p in body_points]
        pygame.gfxdraw.aapolygon(self.screen, body_points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, body_points, self.COLOR_PLAYER)

    def _draw_rotated_polygon(self, pos, points, angle, fill_color, outline_color):
        rotated_points = self._rotate_points(points, angle)
        final_points = [(p[0] + pos.x, p[1] + pos.y) for p in rotated_points]
        
        if len(final_points) > 2:
            pygame.gfxdraw.aapolygon(self.screen, final_points, outline_color)
            pygame.gfxdraw.filled_polygon(self.screen, final_points, fill_color)

    def _rotate_points(self, points, angle):
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in points]

    def _render_ui(self):
        # Time remaining
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_small.render(f"TIME: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Collisions / Lives
        lives_text = self.font_small.render(f"HULL: {self.MAX_COLLISIONS - self.collisions}/{self.MAX_COLLISIONS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(lives_text, (self.WIDTH - lives_text.get_width() - 10, 10))

        # Progress Bar
        bar_y = self.HEIGHT - 20
        progress = self.distance_covered / self.RACE_DISTANCE
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (10, bar_y, self.WIDTH - 20, 10))
        fill_width = max(0, (self.WIDTH - 20) * progress)
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FILL, (10, bar_y, fill_width, 10))

        # Boost Bar
        boost_bar_x = 10
        boost_bar_y = 30
        boost_bar_width = 150
        boost_bar_height = 10
        boost_progress = self.boost_fuel / self.MAX_BOOST_FUEL
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR, (boost_bar_x, boost_bar_y, boost_bar_width, boost_bar_height))
        fill_width = max(0, boost_bar_width * boost_progress)
        pygame.draw.rect(self.screen, self.COLOR_UI_BOOST_FILL, (boost_bar_x, boost_bar_y, fill_width, boost_bar_height))
        
        # Game Over / Win Text
        if self.game_over:
            if self.game_won:
                text = "FINISH!"
                color = self.COLOR_FINISH_LINE
            else:
                text = "GAME OVER"
                color = self.COLOR_BOOST
            
            end_text = self.font_large.render(text, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It requires pygame to be installed with display support.
    # The headless os.environ setting is for the agent, not this block.
    try:
        import pygame
        
        env = GameEnv()
        obs, info = env.reset()
        
        # --- Pygame setup for human play ---
        pygame.display.set_caption("Side-Scrolling Space Racer")
        screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
        clock = pygame.time.Clock()
        running = True

        # --- Game Loop ---
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get keyboard input
            keys = pygame.key.get_pressed()
            
            # Map keys to action space
            movement = 0 # no-op
            if keys[pygame.K_UP]:
                movement = 1
            elif keys[pygame.K_DOWN]:
                movement = 2
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

            action = [movement, space_held, shift_held]
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            if terminated or truncated:
                print(f"Game Over! Score: {info['score']:.2f}, Steps: {info['steps']}")
                pygame.time.wait(3000) # Pause for 3 seconds
                obs, info = env.reset()

            clock.tick(env.FPS)

        env.close()
    except pygame.error as e:
        print("\nCould not run human-playable demo.")
        print("This is expected if you are in a headless environment.")
        print(f"Pygame error: {e}")