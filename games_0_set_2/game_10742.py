import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:58:07.136871
# Source Brief: brief_00742.md
# Brief Index: 742
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    game_description = "Control the strength of two gravity wells to capture falling asteroids and meet the collection targets."
    user_guide = "Controls: Use ←→ arrow keys to adjust the left well's strength and ↑↓ to adjust the right well's strength."
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    DT = 1.0 / FPS
    MAX_STEPS = 1500

    # Colors
    COLOR_BG = (15, 20, 35)
    COLOR_TEXT = (230, 230, 240)
    COLOR_WELL_1 = (0, 150, 255)
    COLOR_WELL_2 = (80, 220, 120)
    ASTEROID_COLORS = [(255, 80, 80), (255, 180, 50), (240, 240, 90)]

    # Physics
    GRAVITY_CONSTANT = 2500
    MIN_WELL_STRENGTH = 0.1
    MAX_WELL_STRENGTH = 5.0
    STRENGTH_INCREMENT = 0.1

    # Game Rules
    WELL_1_TARGET = 10
    WELL_2_TARGET = 5
    DIFFICULTY_RAMP_START_TIME = 30 # in seconds

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20, bold=True)
        
        # Game state variables (defined here for clarity)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        self.spawn_timer = 0.0
        self.initial_spawn_rate = 2.0
        self.spawn_rate = self.initial_spawn_rate

        self.asteroids = []
        self.particles = []

        self.well1 = {}
        self.well2 = {}
        
        # self.reset() is called by the environment wrapper, no need to call it here.

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_elapsed = 0.0
        self.spawn_rate = self.initial_spawn_rate
        self.spawn_timer = self.spawn_rate * 0.5 # Start with a delay

        self.asteroids.clear()
        self.particles.clear()
        
        self.well1 = {
            'pos': np.array([self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT * 0.6]),
            'radius': 25,
            'strength': 1.0,
            'color': self.COLOR_WELL_1,
            'captured_count': 0,
            'target_count': self.WELL_1_TARGET
        }
        self.well2 = {
            'pos': np.array([self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT * 0.6]),
            'radius': 25,
            'strength': 1.0,
            'color': self.COLOR_WELL_2,
            'captured_count': 0,
            'target_count': self.WELL_2_TARGET
        }
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_elapsed += self.DT

        # 1. Handle Actions
        movement, _, _ = action
        if movement == 1:  # Up
            self.well2['strength'] += self.STRENGTH_INCREMENT
        elif movement == 2:  # Down
            self.well2['strength'] -= self.STRENGTH_INCREMENT
        elif movement == 3:  # Left
            self.well1['strength'] -= self.STRENGTH_INCREMENT
        elif movement == 4:  # Right
            self.well1['strength'] += self.STRENGTH_INCREMENT
        
        self.well1['strength'] = np.clip(self.well1['strength'], self.MIN_WELL_STRENGTH, self.MAX_WELL_STRENGTH)
        self.well2['strength'] = np.clip(self.well2['strength'], self.MIN_WELL_STRENGTH, self.MAX_WELL_STRENGTH)

        # 2. Update Game Logic
        self._update_difficulty()
        self._update_spawner()
        
        reward += self._update_asteroids()
        self._update_particles()

        # 3. Check for Termination
        if self.well1['captured_count'] >= self.well1['target_count'] and \
           self.well2['captured_count'] >= self.well2['target_count']:
            reward = 100.0  # Victory reward
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True # Truncated should be True here, but let's stick to terminated for now
            self.game_over = True
            
        if any(a['escaped'] for a in self.asteroids):
            reward = -100.0 # Loss penalty
            terminated = True
            self.game_over = True
            # Keep asteroids on screen for the final frame render
            
        self.score = self.well1['captured_count'] + self.well2['captured_count']
        
        truncated = self.steps >= self.MAX_STEPS and not terminated

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_difficulty(self):
        if self.time_elapsed > self.DIFFICULTY_RAMP_START_TIME:
            # Increase spawn rate by 0.01 per second
            self.spawn_rate = max(0.2, self.spawn_rate - (0.01 * self.DT))

    def _update_spawner(self):
        self.spawn_timer += self.DT
        if self.spawn_timer >= self.spawn_rate:
            self.spawn_timer = 0
            self._create_asteroid()
            
    def _create_asteroid(self):
        radius = self.np_random.integers(10, 15)
        pos = np.array([self.np_random.uniform(radius, self.SCREEN_WIDTH - radius), -radius], dtype=float)
        vel = np.array([0.0, self.np_random.uniform(40, 60)], dtype=float)
        
        num_vertices = self.np_random.integers(5, 9)
        shape_angles = sorted([self.np_random.uniform(0, 2 * math.pi) for _ in range(num_vertices)])
        vertices = []
        for angle in shape_angles:
            r = radius * self.np_random.uniform(0.8, 1.2)
            vertices.append((r * math.cos(angle), r * math.sin(angle)))

        dist_w1 = np.linalg.norm(pos - self.well1['pos'])
        dist_w2 = np.linalg.norm(pos - self.well2['pos'])

        self.asteroids.append({
            'pos': pos,
            'vel': vel,
            'radius': radius,
            'vertices': vertices,
            'color': random.choice(self.ASTEROID_COLORS),
            'angle': 0.0,
            'rot_speed': self.np_random.uniform(-math.pi, math.pi),
            'escaped': False,
            'prev_dist_w1': dist_w1,
            'prev_dist_w2': dist_w2
        })

    def _update_asteroids(self):
        step_reward = 0
        asteroids_to_remove = []

        for i, asteroid in enumerate(self.asteroids):
            # --- Physics Update ---
            force_total = np.array([0.0, 0.0])
            for well in [self.well1, self.well2]:
                vec_to_well = well['pos'] - asteroid['pos']
                dist_sq = np.dot(vec_to_well, vec_to_well)
                if dist_sq > 1: # Avoid extreme forces at point blank
                    force_mag = (self.GRAVITY_CONSTANT * well['strength']) / (dist_sq + 1e-6)
                    force_dir = vec_to_well / np.sqrt(dist_sq)
                    force_total += force_dir * force_mag
            
            asteroid['vel'] += force_total * self.DT
            asteroid['pos'] += asteroid['vel'] * self.DT
            asteroid['angle'] += asteroid['rot_speed'] * self.DT
            
            # --- Reward for getting closer ---
            dist_w1 = np.linalg.norm(asteroid['pos'] - self.well1['pos'])
            dist_w2 = np.linalg.norm(asteroid['pos'] - self.well2['pos'])
            if dist_w1 < asteroid['prev_dist_w1'] or dist_w2 < asteroid['prev_dist_w2']:
                step_reward += 0.01 # Small continuous reward
            asteroid['prev_dist_w1'] = dist_w1
            asteroid['prev_dist_w2'] = dist_w2

            # --- Check for Capture ---
            captured = False
            for well_id, well in enumerate([self.well1, self.well2]):
                if np.linalg.norm(asteroid['pos'] - well['pos']) < well['radius']:
                    well['captured_count'] += 1
                    step_reward += 1.0 # Capture event reward
                    asteroids_to_remove.append(i)
                    self._spawn_particles(asteroid['pos'], well['color'])
                    captured = True
                    break
            if captured:
                continue

            # --- Check for Escape (Loss Condition) ---
            if asteroid['pos'][1] > self.SCREEN_HEIGHT + asteroid['radius']:
                asteroid['escaped'] = True

        # --- Remove captured asteroids ---
        for i in sorted(asteroids_to_remove, reverse=True):
            del self.asteroids[i]

        return step_reward

    def _spawn_particles(self, pos, color):
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(50, 150)
            vel = np.array([math.cos(angle) * speed, math.sin(angle) * speed])
            lifespan = self.np_random.uniform(0.5, 1.5)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'] += p['vel'] * self.DT
            p['vel'] *= 0.95 # Damping
            p['lifespan'] -= self.DT
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def render(self):
        return self._get_observation()

    def _render_game(self):
        self._render_wells()
        self._render_asteroids()
        self._render_particles()

    def _render_wells(self):
        for well in [self.well1, self.well2]:
            # Glow effect
            glow_radius = int(well['radius'] + 20 * (well['strength'] / self.MAX_WELL_STRENGTH))
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (*well['color'], 50), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surface, (int(well['pos'][0] - glow_radius), int(well['pos'][1] - glow_radius)))
            
            # Main circle
            pygame.gfxdraw.aacircle(self.screen, int(well['pos'][0]), int(well['pos'][1]), int(well['radius']), well['color'])
            pygame.gfxdraw.filled_circle(self.screen, int(well['pos'][0]), int(well['pos'][1]), int(well['radius']), well['color'])

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            cos_a = math.cos(asteroid['angle'])
            sin_a = math.sin(asteroid['angle'])
            
            points = []
            for vx, vy in asteroid['vertices']:
                # Rotate
                rotated_x = vx * cos_a - vy * sin_a
                rotated_y = vx * sin_a + vy * cos_a
                # Translate
                screen_x = int(asteroid['pos'][0] + rotated_x)
                screen_y = int(asteroid['pos'][1] + rotated_y)
                points.append((screen_x, screen_y))
            
            if len(points) > 2:
                pygame.gfxdraw.aapolygon(self.screen, points, asteroid['color'])
                pygame.gfxdraw.filled_polygon(self.screen, points, asteroid['color'])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['lifespan'] / p['max_life']))
            color_with_alpha = (*p['color'], alpha)
            radius = int(3 * (p['lifespan'] / p['max_life']))
            if radius > 0:
                pos = (int(p['pos'][0]), int(p['pos'][1]))
                # Use a small surface for alpha blending
                particle_surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surface, color_with_alpha, (radius, radius), radius)
                self.screen.blit(particle_surface, (pos[0] - radius, pos[1] - radius))


    def _render_ui(self):
        # Well 1 UI
        text1 = f"{self.well1['captured_count']}/{self.well1['target_count']}"
        text_surf1 = self.font.render(text1, True, self.COLOR_WELL_1)
        self.screen.blit(text_surf1, (20, 20))
        self._draw_strength_bar(20, 50, self.well1['strength'], self.COLOR_WELL_1)
        
        # Well 2 UI
        text2 = f"{self.well2['captured_count']}/{self.well2['target_count']}"
        text_surf2 = self.font.render(text2, True, self.COLOR_WELL_2)
        text_rect2 = text_surf2.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(text_surf2, text_rect2)
        self._draw_strength_bar(self.SCREEN_WIDTH - 120, 50, self.well2['strength'], self.COLOR_WELL_2)

    def _draw_strength_bar(self, x, y, strength, color):
        bar_width = 100
        bar_height = 10
        fill_ratio = (strength - self.MIN_WELL_STRENGTH) / (self.MAX_WELL_STRENGTH - self.MIN_WELL_STRENGTH)
        fill_width = int(bar_width * fill_ratio)
        
        pygame.draw.rect(self.screen, (50, 50, 60), (x, y, bar_width, bar_height))
        pygame.draw.rect(self.screen, color, (x, y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (x, y, bar_width, bar_height), 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "well1_captured": self.well1['captured_count'],
            "well2_captured": self.well2['captured_count'],
            "asteroids_on_screen": len(self.asteroids)
        }

    def close(self):
        pygame.quit()
        
if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    truncated = False
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Wells")
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print("\n--- Manual Control ---")
    print(GameEnv.user_guide)
    print("R: Reset environment")
    print("Q: Quit")
    
    while running:
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}")
            print(f"Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            truncated = False

        # --- Action Mapping for Manual Play ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        if keys[pygame.K_DOWN]:
            movement = 2
        if keys[pygame.K_LEFT]:
            movement = 3
        if keys[pygame.K_RIGHT]:
            movement = 4
            
        action = [movement, 0, 0] # Space and Shift are not used

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    total_reward = 0
                    terminated = False
                    truncated = False

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)
        
    env.close()