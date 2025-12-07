import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:44:18.533027
# Source Brief: brief_01186.md
# Brief Index: 1186
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for 'Quantum City Escape'.

    The player must gather resources in a procedurally generated quantum city
    while evading quantum police. The goal is to gather enough resources to
    activate an escape route and win.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Gather resources in a procedurally generated quantum city while evading police. "
        "Collect enough resources to activate an escape route and win."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Hold space to activate camouflage (drains resources). "
        "Press shift on an escape route to attempt to win."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 16, 10
        self.CELL_SIZE = 40
        self.FPS = 30
        self.MAX_STEPS = 5000

        # --- Colors (Cyberpunk Neon) ---
        self.COLOR_BG = (10, 0, 20)
        self.COLOR_GRID = (30, 10, 50)
        self.COLOR_PLAYER = (0, 255, 255)
        self.COLOR_POLICE = (255, 0, 100)
        self.COLOR_POLICE_DETECT = (255, 0, 100, 50)
        self.COLOR_RESOURCE = (0, 150, 255)
        self.COLOR_SAFE = (0, 255, 100)
        self.COLOR_ESCAPE = (255, 255, 0)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_ALERT_LOW = (0, 255, 0)
        self.COLOR_ALERT_HIGH = (255, 0, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont('Consolas', 16)
        self.font_m = pygame.font.SysFont('Consolas', 20, bold=True)
        self.font_l = pygame.font.SysFont('Consolas', 48, bold=True)

        # --- Game State Variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.player_grid_pos = None
        self.player_pixel_pos = None
        self.resources = 0
        self.alert_level = 0.0
        self.is_camouflaged = False
        self.city_grid = []
        self.safe_zones = []
        self.escape_routes = []
        self.police = []
        self.particles = []
        self.last_action_feedback = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_city()
        
        start_pos_index = self.np_random.integers(len(self.safe_zones))
        start_pos = self.safe_zones[start_pos_index]
        self.player_grid_pos = np.array(start_pos, dtype=float)
        self.player_pixel_pos = self.player_grid_pos * self.CELL_SIZE + np.array([self.CELL_SIZE / 2, self.CELL_SIZE / 2])

        self.resources = 0
        self.alert_level = 0.0
        self.is_camouflaged = False
        self.police = self._spawn_police()
        self.particles = []
        self.last_action_feedback = {}

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        step_reward = 0
        self.last_action_feedback.clear()

        # --- 1. Unpack Action and Handle Player Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle Movement
        if movement > 0:
            new_pos = self.player_grid_pos.copy()
            if movement == 1: new_pos[1] -= 1  # Up
            elif movement == 2: new_pos[1] += 1  # Down
            elif movement == 3: new_pos[0] -= 1  # Left
            elif movement == 4: new_pos[0] += 1  # Right

            if 0 <= new_pos[0] < self.GRID_W and 0 <= new_pos[1] < self.GRID_H:
                self.player_grid_pos = new_pos
        
        # Handle Camouflage (Space)
        self.is_camouflaged = False
        if space_held and self.resources >= 1:
            self.is_camouflaged = True
            self.resources -= 0.5 # Camo drains resources
            # SFX: Camo activate hum

        # Handle Escape (Shift)
        if shift_held:
            player_cell_type = self.city_grid[int(self.player_grid_pos[1])][int(self.player_grid_pos[0])]['type']
            if player_cell_type == 'escape':
                cost = self.city_grid[int(self.player_grid_pos[1])][int(self.player_grid_pos[0])]['cost']
                if self.resources >= cost:
                    step_reward += 100
                    self.game_over = True
                    self.last_action_feedback['escape'] = 'SUCCESS'
                    # SFX: Victory fanfare
                else:
                    self.last_action_feedback['escape'] = 'FAIL'
                    # SFX: Action denied buzz
        
        # --- 2. Update Game Logic ---
        # Resource Gathering
        current_cell = self.city_grid[int(self.player_grid_pos[1])][int(self.player_grid_pos[0])]
        gathered = current_cell['resource_rate']
        if gathered > 0:
            self.resources += gathered
            step_reward += 0.1 * gathered
            if self.np_random.random() < 0.5: # Spawn particles based on rate
                self._spawn_particles(self.player_pixel_pos, int(gathered * 2))

        # Update Player Pixel Position (Interpolation)
        target_pixel_pos = self.player_grid_pos * self.CELL_SIZE + np.array([self.CELL_SIZE / 2, self.CELL_SIZE / 2])
        self.player_pixel_pos = self.player_pixel_pos * 0.7 + target_pixel_pos * 0.3

        # Update Police
        police_speed_multiplier = 1.0 + (self.steps // 500) * 0.05
        for p in self.police:
            p['pos'] = self._update_entity_pos(p['pos'], p['target'], p['speed'] * police_speed_multiplier)
            if np.linalg.norm(p['pos'] - p['target']) < 5:
                p['target'] = np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)])

        # Update Particles
        self.particles = [particle for particle in self.particles if self._update_particle(particle)]

        # --- 3. Police Detection and Alert Level ---
        old_alert_level = self.alert_level
        detected = False
        min_dist_to_police = float('inf')

        player_cell = self.city_grid[int(self.player_grid_pos[1])][int(self.player_grid_pos[0])]
        in_safe_zone = player_cell['type'] == 'safe'

        for p in self.police:
            dist = np.linalg.norm(self.player_pixel_pos - p['pos'])
            min_dist_to_police = min(min_dist_to_police, dist)
            
            detection_radius = p['detect_radius']
            if self.is_camouflaged: detection_radius *= 0.3
            if in_safe_zone: detection_radius *= 0.1

            if dist < detection_radius:
                detected = True
                break
        
        if detected:
            self.alert_level = min(10.0, self.alert_level + 0.2)
            # SFX: Alert rising tone
        else:
            self.alert_level = max(0.0, self.alert_level - 0.1)

        if self.alert_level > old_alert_level and old_alert_level < 10.0:
            step_reward -= 1.0

        # --- 4. Check Termination Conditions ---
        if self.alert_level >= 10.0:
            step_reward -= 100
            self.game_over = True
            # SFX: Capture siren
        
        terminated = self.game_over
        truncated = self.steps >= self.MAX_STEPS
        if truncated:
             self.game_over = True

        self.score += step_reward
        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _generate_city(self):
        self.city_grid = [[{} for _ in range(self.GRID_W)] for _ in range(self.GRID_H)]
        self.safe_zones = []
        self.escape_routes = []

        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                self.city_grid[y][x] = {'type': 'normal', 'resource_rate': self.np_random.uniform(0.1, 0.5)}

        # Place special zones
        num_safe = self.np_random.integers(2, 5)
        for _ in range(num_safe):
            x, y = self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)
            self.city_grid[y][x] = {'type': 'safe', 'resource_rate': 0}
            self.safe_zones.append((x, y))

        num_escape = self.np_random.integers(1, 3)
        for _ in range(num_escape):
            x, y = self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)
            if (x, y) not in self.safe_zones:
                cost = self.np_random.integers(100, 300)
                self.city_grid[y][x] = {'type': 'escape', 'resource_rate': 0, 'cost': cost}
                self.escape_routes.append((x, y))
        
        if not self.escape_routes: # Anti-softlock
            x, y = self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)
            self.city_grid[y][x] = {'type': 'escape', 'resource_rate': 0, 'cost': 150}
            self.escape_routes.append((x, y))

        num_resource_hubs = self.np_random.integers(5, 10)
        for _ in range(num_resource_hubs):
            x, y = self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)
            if self.city_grid[y][x]['type'] == 'normal':
                self.city_grid[y][x]['resource_rate'] = self.np_random.uniform(1.0, 2.5)

    def _spawn_police(self):
        police_list = []
        num_police = self.np_random.integers(2, 5)
        for _ in range(num_police):
            police_list.append({
                'pos': np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)]),
                'target': np.array([self.np_random.uniform(0, self.WIDTH), self.np_random.uniform(0, self.HEIGHT)]),
                'speed': self.np_random.uniform(0.8, 1.5),
                'detect_radius': self.np_random.uniform(80, 120),
            })
        return police_list

    def _update_entity_pos(self, pos, target, speed):
        direction = target - pos
        norm = np.linalg.norm(direction)
        if norm < speed:
            return target
        velocity = (direction / norm) * speed
        return pos + velocity

    def _spawn_particles(self, pos, amount):
        for _ in range(amount):
            offset = np.array([self.np_random.uniform(-10, 10), self.np_random.uniform(-10, 10)])
            self.particles.append({
                'pos': pos + offset,
                'vel': np.array([self.np_random.uniform(-1, 1), self.np_random.uniform(-1, 1)]),
                'life': self.np_random.uniform(20, 40),
                'color': self.COLOR_RESOURCE
            })

    def _update_particle(self, p):
        p['life'] -= 1
        if p['life'] <= 0: return False
        
        # Move towards player
        attraction = (self.player_pixel_pos - p['pos'])
        norm = np.linalg.norm(attraction)
        if norm > 1:
            attraction = (attraction / norm) * 0.8
        
        p['vel'] = p['vel'] * 0.9 + attraction
        p['pos'] += p['vel']
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_grid()
        self._render_particles()
        self._render_police()
        self._render_player()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "resources": self.resources, "alert_level": self.alert_level}

    def _draw_glow_circle(self, surface, color, center, radius, glow_size):
        for i in range(glow_size, 0, -1):
            alpha = int(color[3] * (1 - i / glow_size))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius + i), (*color[:3], alpha))
        pygame.gfxdraw.filled_circle(surface, int(center[0]), int(center[1]), int(radius), color)
        pygame.gfxdraw.aacircle(surface, int(center[0]), int(center[1]), int(radius), color)

    def _render_background(self):
        for y in range(self.GRID_H + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y * self.CELL_SIZE), (self.WIDTH, y * self.CELL_SIZE))
        for x in range(self.GRID_W + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x * self.CELL_SIZE, 0), (x * self.CELL_SIZE, self.HEIGHT))

    def _render_grid(self):
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                cell = self.city_grid[y][x]
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                color = None
                alpha = 50
                if cell['type'] == 'safe':
                    color = self.COLOR_SAFE
                elif cell['type'] == 'escape':
                    color = self.COLOR_ESCAPE
                elif cell['resource_rate'] > 1.0:
                    color = self.COLOR_RESOURCE
                    alpha = min(100, int(20 * cell['resource_rate']))
                
                if color:
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    pulse = (math.sin(self.steps * 0.1 + x + y) + 1) / 2
                    final_alpha = int(alpha + pulse * 30)
                    s.fill((*color, final_alpha))
                    self.screen.blit(s, rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            life_ratio = max(0, p['life'] / 40.0)
            radius = int(life_ratio * 3)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*p['color'], int(life_ratio * 200)))

    def _render_police(self):
        s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for p in self.police:
            # Detection radius
            player_cell = self.city_grid[int(self.player_grid_pos[1])][int(self.player_grid_pos[0])]
            in_safe_zone = player_cell['type'] == 'safe'
            radius = p['detect_radius']
            if self.is_camouflaged: radius *= 0.3
            if in_safe_zone: radius *= 0.1
            self._draw_glow_circle(s, self.COLOR_POLICE_DETECT, p['pos'], int(radius), 10)

            # Police body
            size = 8
            points = [
                (p['pos'][0], p['pos'][1] - size),
                (p['pos'][0] - size * 0.866, p['pos'][1] + size * 0.5),
                (p['pos'][0] + size * 0.866, p['pos'][1] + size * 0.5),
            ]
            pygame.gfxdraw.aapolygon(s, points, self.COLOR_POLICE)
            pygame.gfxdraw.filled_polygon(s, points, self.COLOR_POLICE)
        self.screen.blit(s, (0, 0))

    def _render_player(self):
        pos = self.player_pixel_pos
        size = 12
        color = self.COLOR_PLAYER

        if self.is_camouflaged:
            # Shimmer effect for camouflage
            s = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pulse = (math.sin(self.steps * 0.5) + 1) / 2
            alpha = int(100 + pulse * 100)
            points = [
                (size, 0),
                (size*2, size),
                (size, size*2),
                (0, size)
            ]
            pygame.gfxdraw.filled_polygon(s, points, (*color, alpha))
            pygame.gfxdraw.aapolygon(s, points, (*color, alpha))
            self.screen.blit(s, (int(pos[0] - size), int(pos[1] - size)))
        else:
            # Normal diamond shape with glow
            points = [
                (pos[0], pos[1] - size),
                (pos[0] + size, pos[1]),
                (pos[0], pos[1] + size),
                (pos[0] - size, pos[1])
            ]
            for i in range(5, 0, -1):
                glow_alpha = int(150 * (1 - i / 5))
                pygame.gfxdraw.filled_polygon(self.screen, points, (*color, glow_alpha))
            
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
            pygame.gfxdraw.aapolygon(self.screen, points, color)

    def _render_ui(self):
        # Resource display
        res_text = self.font_m.render(f"RES: {int(self.resources)}", True, self.COLOR_RESOURCE)
        self.screen.blit(res_text, (10, 10))

        # Score and Steps
        score_text = self.font_s.render(f"SCORE: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        steps_text = self.font_s.render(f"STEP: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(steps_text, (self.WIDTH - steps_text.get_width() - 10, 30))

        # Alert Bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = 10
        alert_ratio = self.alert_level / 10.0
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height))
        
        alert_color = (
            self.COLOR_ALERT_LOW[0] * (1 - alert_ratio) + self.COLOR_ALERT_HIGH[0] * alert_ratio,
            self.COLOR_ALERT_LOW[1] * (1 - alert_ratio) + self.COLOR_ALERT_HIGH[1] * alert_ratio,
            self.COLOR_ALERT_LOW[2] * (1 - alert_ratio) + self.COLOR_ALERT_HIGH[2] * alert_ratio,
        )
        pygame.draw.rect(self.screen, alert_color, (bar_x, bar_y, bar_width * alert_ratio, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)
        alert_label = self.font_s.render("ALERT", True, self.COLOR_TEXT)
        self.screen.blit(alert_label, (bar_x + bar_width / 2 - alert_label.get_width() / 2, bar_y + bar_height))
        
        # Action feedback
        if 'escape' in self.last_action_feedback:
            feedback = self.last_action_feedback['escape']
            if feedback == 'FAIL':
                text = self.font_m.render("INSUFFICIENT RESOURCES", True, self.COLOR_POLICE)
                self.screen.blit(text, (self.player_pixel_pos[0] - text.get_width() / 2, self.player_pixel_pos[1] - 40))

        # Game Over Screen
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            if self.alert_level >= 10.0:
                text = self.font_l.render("CAPTURED", True, self.COLOR_POLICE)
            elif self.steps >= self.MAX_STEPS:
                text = self.font_l.render("TIME OUT", True, self.COLOR_ESCAPE)
            else: # Escaped
                text = self.font_l.render("ESCAPED", True, self.COLOR_SAFE)
            
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            s.blit(text, text_rect)
            self.screen.blit(s, (0,0))

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually
    # You need to unset the dummy video driver to see the window
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    total_reward = 0
    
    # Create a display for manual play
    pygame.display.set_caption("Quantum City Escape")
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]: movement = 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: movement = 2
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: movement = 3
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

    env.close()