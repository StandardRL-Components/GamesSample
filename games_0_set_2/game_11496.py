import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T13:03:27.433862
# Source Brief: brief_01496.md
# Brief Index: 1496
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
        "Control two gravity wells to capture asteroids and fill them to capacity. "
        "Once one well is nearly full, a timer starts, and you must fill both to win."
    )
    user_guide = (
        "Use W/S to control the left well's strength and ↑/↓ arrows for the right well. "
        "Capture asteroids to fill the wells."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000
    WIN_CAPACITY_THRESHOLD = 75.0
    TIMER_DURATION_SECONDS = 15

    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_STAR = (100, 100, 120)
    COLOR_ASTEROID = (180, 180, 180)
    COLOR_WELL_BLUE = (50, 150, 255)
    COLOR_WELL_BLUE_FILL = (100, 200, 255)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_TIMER_GREEN = (100, 255, 100)
    COLOR_TIMER_YELLOW = (255, 255, 100)
    COLOR_TIMER_RED = (255, 100, 100)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_timer = pygame.font.SysFont("monospace", 32, bold=True)
        self.font_well = pygame.font.SysFont("monospace", 16, bold=True)

        # --- State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer_active = False
        self.timer = 0
        self.level = 1
        
        self.left_well = {}
        self.right_well = {}
        self.asteroids = []
        self.particles = []
        self.stars = []

        self.last_left_capacity_milestone = 0
        self.last_right_capacity_milestone = 0

        # --- Game Parameters ---
        self.well_pos_left = (self.SCREEN_WIDTH * 0.25, self.SCREEN_HEIGHT / 2)
        self.well_pos_right = (self.SCREEN_WIDTH * 0.75, self.SCREEN_HEIGHT / 2)
        self.well_radius = 65
        self.gravity_constant = 7000
        self.max_asteroids = 15
        self.asteroid_spawn_chance = 0.08
        self.asteroid_base_speed = 0.8

        self._generate_stars()
        # self.reset() is called by the wrapper, no need to call it here.
        
        # --- Critical Self-Check ---
        # self.validate_implementation() # This can be removed in final version


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0.0
        self.game_over = False
        self.timer_active = False
        self.timer = self.TIMER_DURATION_SECONDS * self.FPS
        self.level = 1

        self.left_well = {
            'pos': self.well_pos_left, 'strength': 50.0, 'capacity': 0.0,
            'radius': self.well_radius, 'color': self.COLOR_WELL_BLUE, 'fill_color': self.COLOR_WELL_BLUE_FILL
        }
        self.right_well = {
            'pos': self.well_pos_right, 'strength': 50.0, 'capacity': 0.0,
            'radius': self.well_radius, 'color': self.COLOR_WELL_BLUE, 'fill_color': self.COLOR_WELL_BLUE_FILL
        }

        self.last_left_capacity_milestone = 0
        self.last_right_capacity_milestone = 0

        self.asteroids = []
        self.particles = []
        for _ in range(self.max_asteroids // 2):
            self._spawn_asteroid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        # --- 1. Handle Actions ---
        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        strength_change = 2.5 # Percentage points
        if movement == 1: # Up: Left well strength up
            self.left_well['strength'] += strength_change
        elif movement == 2: # Down: Left well strength down
            self.left_well['strength'] -= strength_change
        elif movement == 3: # Left: Right well strength down
            self.right_well['strength'] -= strength_change
        elif movement == 4: # Right: Right well strength up
            self.right_well['strength'] += strength_change
        
        self.left_well['strength'] = np.clip(self.left_well['strength'], 0, 100)
        self.right_well['strength'] = np.clip(self.right_well['strength'], 0, 100)

        # --- 2. Update Game Logic ---
        self._update_asteroids()
        self._update_particles()
        self._spawn_asteroid_if_needed()

        # --- 3. Calculate Reward ---
        # Note: Capture rewards are handled in _update_asteroids
        reward += self._check_capacity_milestones()
        self.score += reward

        # --- 4. Update Timer & Level ---
        if not self.timer_active:
            if self.left_well['capacity'] >= self.WIN_CAPACITY_THRESHOLD or self.right_well['capacity'] >= self.WIN_CAPACITY_THRESHOLD:
                self.timer_active = True
                self.level = 2 # Trigger difficulty increase
        
        if self.timer_active:
            self.timer -= 1
        
        # --- 5. Check Termination ---
        terminated = False
        truncated = False
        win = self.left_well['capacity'] >= 100.0 and self.right_well['capacity'] >= 100.0
        
        if win:
            reward += 100.0
            self.score += 100.0
            terminated = True
            # sound: win_sound
        elif self.timer_active and self.timer <= 0:
            reward -= 100.0
            self.score -= 100.0
            terminated = True
            # sound: lose_sound
        elif self.steps >= self.MAX_STEPS:
            truncated = True
        
        if terminated or truncated:
            self.game_over = True

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # --- Update Methods ---

    def _update_asteroids(self):
        asteroids_to_remove = []
        for i, asteroid in enumerate(self.asteroids):
            # Apply gravity from both wells
            self._apply_gravity(asteroid, self.left_well)
            self._apply_gravity(asteroid, self.right_well)
            
            # Update position
            asteroid['pos'][0] += asteroid['vel'][0]
            asteroid['pos'][1] += asteroid['vel'][1]

            # Check for capture
            captured_by = None
            if self._is_captured(asteroid, self.left_well):
                captured_by = self.left_well
            elif self._is_captured(asteroid, self.right_well):
                captured_by = self.right_well

            if captured_by:
                # sound: capture_pop
                captured_by['capacity'] = min(100.0, captured_by['capacity'] + asteroid['mass'])
                self.score += 0.1 # Continuous reward for capture
                self._create_particles(asteroid['pos'], captured_by['fill_color'])
                asteroids_to_remove.append(i)
                continue

            # Remove if off-screen
            if not ((-asteroid['size'] < asteroid['pos'][0] < self.SCREEN_WIDTH + asteroid['size']) and \
                    (-asteroid['size'] < asteroid['pos'][1] < self.SCREEN_HEIGHT + asteroid['size'])):
                asteroids_to_remove.append(i)

        # Remove asteroids in reverse order to avoid index issues
        for i in sorted(asteroids_to_remove, reverse=True):
            del self.asteroids[i]

    def _apply_gravity(self, asteroid, well):
        if well['strength'] == 0:
            return
            
        dx = well['pos'][0] - asteroid['pos'][0]
        dy = well['pos'][1] - asteroid['pos'][1]
        dist_sq = dx*dx + dy*dy
        
        # Avoid division by zero and extreme forces at close range
        if dist_sq < 100:
            dist_sq = 100

        dist = math.sqrt(dist_sq)
        force = (self.gravity_constant * well['strength'] / 100.0) / dist_sq
        
        ax = force * dx / dist
        ay = force * dy / dist
        
        asteroid['vel'][0] += ax
        asteroid['vel'][1] += ay

    def _is_captured(self, asteroid, well):
        dx = well['pos'][0] - asteroid['pos'][0]
        dy = well['pos'][1] - asteroid['pos'][1]
        dist_sq = dx*dx + dy*dy
        return dist_sq < (well['radius'] * 0.9)**2

    def _check_capacity_milestones(self):
        reward = 0.0
        
        current_left_milestone = int(self.left_well['capacity'] // 10)
        if current_left_milestone > self.last_left_capacity_milestone:
            reward += 1.0 * (current_left_milestone - self.last_left_capacity_milestone)
            self.last_left_capacity_milestone = current_left_milestone
            # sound: milestone_achieved

        current_right_milestone = int(self.right_well['capacity'] // 10)
        if current_right_milestone > self.last_right_capacity_milestone:
            reward += 1.0 * (current_right_milestone - self.last_right_capacity_milestone)
            self.last_right_capacity_milestone = current_right_milestone
            # sound: milestone_achieved
            
        return reward

    # --- Spawning Methods ---

    def _generate_stars(self):
        self.stars = []
        for _ in range(150):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.choice([1, 1, 1, 2])
            self.stars.append({'pos': (x, y), 'size': size})

    def _spawn_asteroid_if_needed(self):
        if len(self.asteroids) < self.max_asteroids and self.np_random.random() < self.asteroid_spawn_chance:
            self._spawn_asteroid()

    def _spawn_asteroid(self):
        edge = self.np_random.choice(['top', 'bottom', 'left', 'right'])
        
        if edge == 'left':
            pos = [-20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
            vel = [self.np_random.uniform(0.5, 1.5), self.np_random.uniform(-0.5, 0.5)]
        elif edge == 'right':
            pos = [self.SCREEN_WIDTH + 20, self.np_random.uniform(0, self.SCREEN_HEIGHT)]
            vel = [self.np_random.uniform(-1.5, -0.5), self.np_random.uniform(-0.5, 0.5)]
        elif edge == 'top':
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), -20]
            vel = [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(0.5, 1.5)]
        else: # bottom
            pos = [self.np_random.uniform(0, self.SCREEN_WIDTH), self.SCREEN_HEIGHT + 20]
            vel = [self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-1.5, -0.5)]
        
        speed_multiplier = self.asteroid_base_speed + (self.level - 1) * 0.5
        vel = [v * speed_multiplier for v in vel]

        size = self.np_random.uniform(5, 12)
        mass = (size / 12.0) * 5.0 # Mass scales with size, max mass is 5
        
        num_points = self.np_random.integers(5, 9)
        points = []
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            radius = size * self.np_random.uniform(0.8, 1.2)
            points.append((math.cos(angle) * radius, math.sin(angle) * radius))

        self.asteroids.append({'pos': pos, 'vel': vel, 'size': size, 'mass': mass, 'points': points})
    
    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            size = self.np_random.uniform(2, 5)
            self.particles.append({'pos': list(pos), 'vel': vel, 'size': size, 'life': 20, 'color': color})

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # Damping
            p['vel'][1] *= 0.95
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    # --- Rendering Methods ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_asteroids()
        self._render_wells()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, star['pos'], star['size'])

    def _render_wells(self):
        self._render_single_well(self.left_well)
        self._render_single_well(self.right_well)
        
    def _render_single_well(self, well):
        pos = (int(well['pos'][0]), int(well['pos'][1]))
        
        # Draw capacity fill
        fill_radius = int(well['radius'] * math.sqrt(well['capacity'] / 100.0))
        if fill_radius > 0:
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], fill_radius, well['fill_color'])
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], fill_radius, well['fill_color'])

        # Draw gravity field glow
        glow_strength = int(255 * (well['strength'] / 100.0))
        for i in range(5):
            alpha = int(glow_strength * (1 - i/5) * 0.2)
            color = (well['color'][0], well['color'][1], well['color'][2], alpha)
            radius = well['radius'] + i * 3
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, color)
        
        # Draw main outline
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], well['radius'], well['color'])

    def _render_asteroids(self):
        for asteroid in self.asteroids:
            points = [(p[0] + asteroid['pos'][0], p[1] + asteroid['pos'][1]) for p in asteroid['points']]
            int_points = [(int(p[0]), int(p[1])) for p in points]
            if len(int_points) > 2:
                pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_ASTEROID)
                pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_ASTEROID)

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 20.0))
            color = (p['color'][0], p['color'][1], p['color'][2], alpha)
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            size = int(p['size'] * (p['life'] / 20.0))
            if size > 0:
                pygame.draw.circle(self.screen, color, pos, size)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        if self.timer_active:
            seconds = self.timer / self.FPS
            timer_color = self.COLOR_TIMER_GREEN
            if seconds < self.TIMER_DURATION_SECONDS * 0.66:
                timer_color = self.COLOR_TIMER_YELLOW
            if seconds < self.TIMER_DURATION_SECONDS * 0.33:
                timer_color = self.COLOR_TIMER_RED
            
            timer_text = self.font_timer.render(f"{seconds:.1f}", True, timer_color)
            text_rect = timer_text.get_rect(center=(self.SCREEN_WIDTH / 2, 25))
            self.screen.blit(timer_text, text_rect)

        # Well Info
        self._render_well_ui(self.left_well)
        self._render_well_ui(self.right_well)
        
    def _render_well_ui(self, well):
        pos = well['pos']
        # Capacity Text
        cap_text = self.font_main.render(f"{well['capacity']:.0f}%", True, self.COLOR_UI_TEXT)
        cap_rect = cap_text.get_rect(center=(pos[0], pos[1]))
        self.screen.blit(cap_text, cap_rect)

        # Strength Bar
        bar_width = 100
        bar_height = 10
        bar_x = pos[0] - bar_width / 2
        bar_y = pos[1] + well['radius'] + 15
        
        fill_width = (well['strength'] / 100.0) * bar_width
        pygame.draw.rect(self.screen, (50,50,70), (bar_x, bar_y, bar_width, bar_height), border_radius=3)
        if fill_width > 0:
            pygame.draw.rect(self.screen, well['color'], (bar_x, bar_y, fill_width, bar_height), border_radius=3)
        
        str_text = self.font_well.render("STRENGTH", True, self.COLOR_UI_TEXT)
        str_rect = str_text.get_rect(center=(pos[0], bar_y + bar_height + 10))
        self.screen.blit(str_text, str_rect)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "left_well_capacity": self.left_well['capacity'],
            "right_well_capacity": self.right_well['capacity'],
            "timer": self.timer / self.FPS if self.timer_active else -1,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It will not run with the dummy video driver, so we unset it.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gravity Wells")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        # --- Action Mapping for Manual Play ---
        # 0=none, 1=up, 2=down, 3=left, 4=right
        action = [0, 0, 0] # Default no-op
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w]:
            action[0] = 1 # Left well strength up
        elif keys[pygame.K_s]:
            action[0] = 2 # Left well strength down
            
        if keys[pygame.K_UP]:
            action[0] = 4 # Right well strength up
        elif keys[pygame.K_DOWN]:
            action[0] = 3 # Right well strength down

        # Use A/D for both wells
        if keys[pygame.K_d]:
             action[0] = 4
        elif keys[pygame.K_a]:
             action[0] = 3

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
    
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Info: {info}")
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(env.FPS)
        
    env.close()