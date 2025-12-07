import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T11:27:57.777665
# Source Brief: brief_00488.md
# Brief Index: 488
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = (
        "Navigate a spaceship through an asteroid field, collecting enough energy cells to win before time runs out."
    )
    user_guide = (
        "Controls: Use ↑ to thrust forward, ↓ for reverse thrust, and ←→ to rotate your ship."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 60  # Target FPS for rendering and game logic
        self.TIME_LIMIT_SECONDS = 60
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Player settings
        self.SHIP_THRUST = 0.1
        self.SHIP_REVERSE_THRUST = 0.05
        self.SHIP_ROTATION_SPEED = 4.0
        self.SHIP_DRAG = 0.985
        self.SHIP_MAX_SPEED = 5.0
        self.SHIP_RADIUS = 12

        # Entity counts
        self.INITIAL_ASTEROIDS = 10
        self.INITIAL_CELLS = 5
        self.CELLS_TO_TRIGGER_SPAWN = 3
        self.CELLS_TO_SPAWN = 2

        # Colors
        self.COLOR_BG = (10, 15, 30)
        self.COLOR_PLAYER = (255, 255, 255)
        self.COLOR_THRUST = (255, 180, 80)
        self.COLOR_ASTEROID = (120, 130, 140)
        self.COLOR_CELL = (0, 200, 255)
        self.COLOR_CELL_GLOW = (0, 150, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_BAR_BG = (50, 50, 80)
        self.COLOR_ENERGY_BAR = (0, 200, 255)
        self.COLOR_TIME_OK = (0, 255, 150)
        self.COLOR_TIME_WARN = (255, 80, 80)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_remaining = None
        self.energy = None
        self.cells_collected_since_spawn = None

        self.ship_pos = None
        self.ship_vel = None
        self.ship_angle = None

        self.asteroids = None
        self.energy_cells = None
        self.particles = None
        self.last_closest_cell_dist = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.energy = 0
        self.cells_collected_since_spawn = 0

        self.ship_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        self.ship_vel = pygame.Vector2(0, 0)
        self.ship_angle = -90  # Pointing up

        self.particles = []
        
        safe_spawn_radius = 100
        self.asteroids = [self._spawn_entity(safe_spawn_radius) for _ in range(self.INITIAL_ASTEROIDS)]
        self.energy_cells = [self._spawn_entity(safe_spawn_radius) for _ in range(self.INITIAL_CELLS)]
        
        self.last_closest_cell_dist = self._get_closest_cell_dist()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, _, _ = action  # space and shift are unused
        reward = 0
        
        # 1. Update game logic
        self._handle_input(movement)
        self._update_physics()

        # 2. Check for events and calculate rewards
        # Cell collection
        collected_cell = self._check_cell_collisions()
        if collected_cell:
            reward += 10.0  # +10 for collecting a cell
            self.score += 10
            self.energy += 1
            self.cells_collected_since_spawn += 1
            if self.cells_collected_since_spawn >= self.CELLS_TO_TRIGGER_SPAWN:
                self.cells_collected_since_spawn = 0
                for _ in range(self.CELLS_TO_SPAWN):
                    self.energy_cells.append(self._spawn_entity(0))
            # sound_placeholder: "pickup.wav"

        # Continuous reward for getting closer to a cell
        closest_dist = self._get_closest_cell_dist()
        if self.last_closest_cell_dist is not None and closest_dist is not None:
            distance_diff = self.last_closest_cell_dist - closest_dist
            reward += distance_diff * 0.1 # Small reward for closing distance
        self.last_closest_cell_dist = closest_dist

        # 3. Check for termination conditions
        terminated = False
        truncated = False
        if self._check_asteroid_collision():
            reward = -100.0  # -100 for crashing
            self.score -= 100
            terminated = True
            self.game_over = True
            # sound_placeholder: "explosion.wav"

        if self.energy >= 10:
            reward = 100.0  # +100 for winning
            self.score += 100
            terminated = True
            self.game_over = True
            # sound_placeholder: "win.wav"

        self.time_remaining -= 1
        self.steps += 1
        if self.time_remaining <= 0:
            if not terminated: # Avoid double penalty if crashed on last frame
                reward = -100.0  # -100 for timeout
                self.score -= 100
            terminated = True
            self.game_over = True
            # sound_placeholder: "timeout.wav"
            
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _spawn_entity(self, safe_radius_from_center):
        center = pygame.Vector2(self.WIDTH / 2, self.HEIGHT / 2)
        while True:
            pos = pygame.Vector2(
                self.np_random.uniform(20, self.WIDTH - 20),
                self.np_random.uniform(20, self.HEIGHT - 20)
            )
            if pos.distance_to(center) > safe_radius_from_center:
                return pos

    def _handle_input(self, movement):
        # movement: 0=none, 1=up, 2=down, 3=left, 4=right
        if movement == 1:  # Thrust forward
            thrust_vec = pygame.Vector2(math.cos(math.radians(self.ship_angle)), math.sin(math.radians(self.ship_angle))) * self.SHIP_THRUST
            self.ship_vel += thrust_vec
            self._create_thrust_particles()
        elif movement == 2:  # Reverse thrust
            thrust_vec = pygame.Vector2(math.cos(math.radians(self.ship_angle)), math.sin(math.radians(self.ship_angle))) * self.SHIP_REVERSE_THRUST
            self.ship_vel -= thrust_vec
        if movement == 3:  # Rotate left
            self.ship_angle -= self.SHIP_ROTATION_SPEED
        elif movement == 4:  # Rotate right
            self.ship_angle += self.SHIP_ROTATION_SPEED

    def _update_physics(self):
        # Cap speed
        if self.ship_vel.length() > self.SHIP_MAX_SPEED:
            self.ship_vel.scale_to_length(self.SHIP_MAX_SPEED)

        # Apply drag
        self.ship_vel *= self.SHIP_DRAG
        
        # Update position
        self.ship_pos += self.ship_vel

        # Screen wrap
        self.ship_pos.x %= self.WIDTH
        self.ship_pos.y %= self.HEIGHT
        
        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _check_cell_collisions(self):
        for cell_pos in self.energy_cells[:]:
            if self.ship_pos.distance_to(cell_pos) < self.SHIP_RADIUS + 10:
                self.energy_cells.remove(cell_pos)
                self._create_collection_particles(cell_pos)
                return True
        return False

    def _check_asteroid_collision(self):
        for ast_pos in self.asteroids:
            if self.ship_pos.distance_to(ast_pos) < self.SHIP_RADIUS + 15: # 15 is avg asteroid radius
                self._create_explosion_particles(self.ship_pos)
                return True
        return False
        
    def _get_closest_cell_dist(self):
        if not self.energy_cells:
            return None
        return min(self.ship_pos.distance_to(cell) for cell in self.energy_cells)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_asteroids()
        self._render_cells()
        self._render_player()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "energy": self.energy, "time_left": self.time_remaining / self.FPS}

    def _render_stars(self):
        # Static starfield based on a seed for consistency
        star_seed = 12345
        st_rand = random.Random(star_seed)
        for _ in range(150):
            x = st_rand.randint(0, self.WIDTH)
            y = st_rand.randint(0, self.HEIGHT)
            size = st_rand.choice([1, 1, 1, 2])
            brightness = st_rand.randint(50, 100)
            self.screen.set_at((x, y), (brightness, brightness, brightness))

    def _render_particles(self):
        for p in self.particles:
            alpha = p['life'] / p['max_life']
            color = tuple(c * alpha for c in p['color'])
            pygame.draw.circle(self.screen, color, p['pos'], int(p['size'] * alpha))

    def _render_asteroids(self):
        for pos in self.asteroids:
            self._draw_aa_polygon(self.screen, self._get_asteroid_shape(pos), self.COLOR_ASTEROID)

    def _render_cells(self):
        for pos in self.energy_cells:
            # Pulsating glow
            glow_size = 15 + 4 * math.sin(self.steps * 0.1)
            glow_alpha = 100 + 50 * math.sin(self.steps * 0.1)
            s = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_CELL_GLOW, glow_alpha), (glow_size, glow_size), glow_size)
            self.screen.blit(s, (int(pos.x - glow_size), int(pos.y - glow_size)), special_flags=pygame.BLEND_RGBA_ADD)
            
            # Core circle
            pygame.gfxdraw.aacircle(self.screen, int(pos.x), int(pos.y), 8, self.COLOR_CELL)
            pygame.gfxdraw.filled_circle(self.screen, int(pos.x), int(pos.y), 8, self.COLOR_CELL)

    def _render_player(self):
        # Don't draw player if they crashed into an asteroid
        is_crashed = False
        for ast_pos in self.asteroids:
            if self.ship_pos.distance_to(ast_pos) < self.SHIP_RADIUS + 15:
                is_crashed = True
                break
        if self.game_over and is_crashed:
            return

        angle_rad = math.radians(self.ship_angle)
        p1 = self.ship_pos + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * self.SHIP_RADIUS
        p2 = self.ship_pos + pygame.Vector2(math.cos(angle_rad + 2.5), math.sin(angle_rad + 2.5)) * self.SHIP_RADIUS
        p3 = self.ship_pos + pygame.Vector2(math.cos(angle_rad - 2.5), math.sin(angle_rad - 2.5)) * self.SHIP_RADIUS
        
        self._draw_aa_polygon(self.screen, [p1, p2, p3], self.COLOR_PLAYER)

    def _render_ui(self):
        # Energy Bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = 10
        energy_ratio = min(1.0, self.energy / 10.0)
        
        pygame.draw.rect(self.screen, self.COLOR_BAR_BG, (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        pygame.draw.rect(self.screen, self.COLOR_ENERGY_BAR, (bar_x, bar_y, bar_width * energy_ratio, bar_height), border_radius=3)
        
        # Timer
        time_sec = max(0, self.time_remaining / self.FPS)
        time_text = f"TIME: {time_sec:.1f}"
        time_color = self.COLOR_TIME_OK if time_sec > 10 else self.COLOR_TIME_WARN
        text_surface = self.font_small.render(time_text, True, time_color)
        self.screen.blit(text_surface, (self.WIDTH - text_surface.get_width() - 15, 10))
        
        # Energy Text
        energy_text = "ENERGY"
        energy_surface = self.font_small.render(energy_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(energy_surface, (bar_x - energy_surface.get_width() - 10, 8))

    def _render_game_over(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.energy >= 10:
            msg = "MISSION COMPLETE"
            color = self.COLOR_ENERGY_BAR
        else:
            msg = "GAME OVER"
            color = self.COLOR_TIME_WARN

        text_surf = self.font_large.render(msg, True, color)
        text_rect = text_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        overlay.blit(text_surf, text_rect)
        self.screen.blit(overlay, (0,0))
        
    def _get_asteroid_shape(self, pos):
        # Use position as seed for consistent asteroid shapes
        ast_seed = int(pos.x * 100 + pos.y)
        ast_rand = random.Random(ast_seed)
        num_points = ast_rand.randint(7, 10)
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dist = ast_rand.uniform(12, 18)
            points.append(pos + pygame.Vector2(math.cos(angle), math.sin(angle)) * dist)
        return points

    def _draw_aa_polygon(self, surface, points, color):
        pygame.gfxdraw.aapolygon(surface, [(int(p.x), int(p.y)) for p in points], color)
        pygame.gfxdraw.filled_polygon(surface, [(int(p.x), int(p.y)) for p in points], color)
        
    def _create_thrust_particles(self):
        angle_rad = math.radians(self.ship_angle + 180) # Opposite direction of thrust
        for _ in range(2):
            vel_angle = angle_rad + self.np_random.uniform(-0.3, 0.3)
            vel_mag = self.np_random.uniform(1, 2.5)
            p_vel = pygame.Vector2(math.cos(vel_angle), math.sin(vel_angle)) * vel_mag
            self.particles.append({
                'pos': self.ship_pos - pygame.Vector2(math.cos(math.radians(self.ship_angle)), math.sin(math.radians(self.ship_angle))) * 10,
                'vel': p_vel + self.ship_vel * 0.5,
                'life': self.np_random.integers(15, 25),
                'max_life': 25,
                'color': self.COLOR_THRUST,
                'size': self.np_random.uniform(1, 3)
            })

    def _create_collection_particles(self, pos):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(1, 3)
            p_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * vel_mag
            self.particles.append({
                'pos': pos.copy(),
                'vel': p_vel,
                'life': self.np_random.integers(20, 40),
                'max_life': 40,
                'color': self.COLOR_CELL,
                'size': self.np_random.uniform(1, 4)
            })

    def _create_explosion_particles(self, pos):
        for _ in range(50):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel_mag = self.np_random.uniform(1, 5)
            p_vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * vel_mag
            self.particles.append({
                'pos': pos.copy(),
                'vel': p_vel,
                'life': self.np_random.integers(30, 60),
                'max_life': 60,
                'color': self.COLOR_PLAYER if self.np_random.random() > 0.33 else (self.COLOR_THRUST if self.np_random.random() > 0.5 else self.COLOR_ASTEROID),
                'size': self.np_random.uniform(2, 5)
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # --- Manual Play Example ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Un-dummy the video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    pygame.display.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Asteroid Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        
        # Handle rotation and thrust separately to allow combined inputs
        action_movement = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action_movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action_movement = 4
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            action_movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action_movement = 2
            
        action = [action_movement, 0, 0] # Space/Shift not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()