import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:15:51.610762
# Source Brief: brief_00984.md
# Brief Index: 984
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment where the player guides a colorful comet through a nebula.
    The goal is to collect energy by matching the comet's color with energy nodes,
    while avoiding obstacles. The player can control the comet's launch trajectory
    and activate a time-slowing ability during flight.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = "Guide a comet through a nebula to collect matching-colored energy nodes while avoiding obstacles."
    user_guide = "Aim with ←→ and ↑↓ to set power. Press space to launch. Hold shift in flight to slow time."
    auto_advance = True

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    WIN_SCORE = 1000
    MAX_STEPS = 2000
    FPS = 30
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_OBSTACLE = (80, 80, 90)
    COLOR_PALETTE = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
    ]
    COLOR_WHITE = (240, 240, 240)
    COLOR_TIME_SLOW_OVERLAY = (100, 150, 255, 40)

    # Physics & Gameplay
    GRAVITY = 0.15
    LAUNCH_POWER_MIN = 3.0
    LAUNCH_POWER_MAX = 15.0
    LAUNCH_ANGLE_MIN = -160
    LAUNCH_ANGLE_MAX = -20
    COMET_RADIUS = 10
    NODE_RADIUS = 15
    TIME_SLOW_FACTOR = 0.4
    TIME_SLOW_MAX_FUEL = 100.0
    TIME_SLOW_CONSUMPTION = 0.8
    TIME_SLOW_REGEN = 0.2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # State variables (initialized in reset)
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_phase = None # "AIMING" or "LAUNCHED"
        
        self.comet_pos = None
        self.comet_vel = None
        self.comet_color = None
        self.comet_trail = None
        
        self.launch_angle = None
        self.launch_power = None
        self.prev_space_held = None
        
        self.time_slow_active = None
        self.time_slow_fuel = None
        
        self.stars = []
        self.nodes = []
        self.obstacles = []
        self.particles = []
        
        self.difficulty_level = 0
        self.last_dist_to_node = float('inf')
        self.last_dist_to_obstacle = float('inf')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_phase = "AIMING"
        
        start_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 30)
        self.comet_pos = start_pos
        self.comet_vel = pygame.Vector2(0, 0)
        self.comet_color = self.COLOR_PALETTE[self.np_random.integers(0, len(self.COLOR_PALETTE))]
        self.comet_trail = []
        
        self.launch_angle = -90.0
        self.launch_power = (self.LAUNCH_POWER_MIN + self.LAUNCH_POWER_MAX) / 2
        self.prev_space_held = False
        
        self.time_slow_active = False
        self.time_slow_fuel = self.TIME_SLOW_MAX_FUEL
        
        self.particles = []
        self._generate_level()

        self.last_dist_to_node = self._get_closest_dist('node')
        self.last_dist_to_obstacle = self._get_closest_dist('obstacle')

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0.0
        
        # --- Update game logic based on phase ---
        if self.game_phase == "AIMING":
            self._update_aiming(movement)
            space_pressed = space_held and not self.prev_space_held
            if space_pressed:
                self._launch_comet()
                # SFX: Launch sound
        
        elif self.game_phase == "LAUNCHED":
            reward += self._update_flight(shift_held)

        # Update previous action state for edge detection
        self.prev_space_held = space_held

        # --- Calculate continuous rewards ---
        if self.game_phase == "LAUNCHED":
            dist_to_node = self._get_closest_dist('node')
            if dist_to_node < self.last_dist_to_node:
                reward += 0.1
            self.last_dist_to_node = dist_to_node

            dist_to_obstacle = self._get_closest_dist('obstacle')
            if dist_to_obstacle < self.last_dist_to_obstacle and dist_to_obstacle < 100:
                reward -= 0.1
            self.last_dist_to_obstacle = dist_to_obstacle

        # --- Check for termination conditions ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if terminated and not self.game_over: # Win condition
            reward += 100
            self.game_over = True
        elif terminated and self.game_over: # Lose condition
            reward -= 10

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_aiming(self, movement):
        if movement == 1: # Up
            self.launch_power = min(self.LAUNCH_POWER_MAX, self.launch_power + 0.2)
        elif movement == 2: # Down
            self.launch_power = max(self.LAUNCH_POWER_MIN, self.launch_power - 0.2)
        elif movement == 3: # Left
            self.launch_angle = max(self.LAUNCH_ANGLE_MIN, self.launch_angle - 1.5)
        elif movement == 4: # Right
            self.launch_angle = min(self.LAUNCH_ANGLE_MAX, self.launch_angle + 1.5)

    def _launch_comet(self):
        self.game_phase = "LAUNCHED"
        angle_rad = math.radians(self.launch_angle)
        self.comet_vel = pygame.Vector2(
            math.cos(angle_rad) * self.launch_power,
            math.sin(angle_rad) * self.launch_power
        )
        # SFX: whoosh_launch.wav

    def _update_flight(self, shift_held):
        reward = 0
        
        # --- Time Slow ---
        if shift_held and self.time_slow_fuel > 0:
            self.time_slow_active = True
            self.time_slow_fuel = max(0, self.time_slow_fuel - self.TIME_SLOW_CONSUMPTION)
        else:
            self.time_slow_active = False
            self.time_slow_fuel = min(self.TIME_SLOW_MAX_FUEL, self.time_slow_fuel + self.TIME_SLOW_REGEN)

        time_factor = self.TIME_SLOW_FACTOR if self.time_slow_active else 1.0

        # --- Physics & Movement ---
        self.comet_vel.y += self.GRAVITY * time_factor
        self.comet_pos += self.comet_vel * time_factor
        
        self.comet_trail.append(pygame.Vector2(self.comet_pos))
        if len(self.comet_trail) > 50:
            self.comet_trail.pop(0)

        # --- Collision Detection ---
        # Screen boundaries
        if self.comet_pos.x < self.COMET_RADIUS or self.comet_pos.x > self.WIDTH - self.COMET_RADIUS:
            self.comet_vel.x *= -0.8
            self.comet_pos.x = np.clip(self.comet_pos.x, self.COMET_RADIUS, self.WIDTH - self.COMET_RADIUS)
            # SFX: bounce.wav
        if self.comet_pos.y < self.COMET_RADIUS:
            self.comet_vel.y *= -0.8
            self.comet_pos.y = self.COMET_RADIUS
            # SFX: bounce.wav
        if self.comet_pos.y > self.HEIGHT + self.COMET_RADIUS * 2: # Fell off bottom
            self.game_over = True
        
        # Obstacles
        for obst in self.obstacles:
            if obst.collidepoint(self.comet_pos):
                self.game_over = True
                self._spawn_particles(self.comet_pos, self.COLOR_OBSTACLE, 50, 4)
                # SFX: explosion.wav
                break
        
        # Nodes
        for node in self.nodes:
            if not node['collected']:
                dist = self.comet_pos.distance_to(node['pos'])
                if dist < self.COMET_RADIUS + self.NODE_RADIUS:
                    node['collected'] = True
                    if tuple(node['color']) == tuple(self.comet_color):
                        self.score += 50
                        reward += 1.0
                        self._spawn_particles(node['pos'], node['color'], 30, 2)
                        # SFX: collect_good.wav
                    else:
                        self.score += 10
                        self._spawn_particles(node['pos'], (128,128,128), 20, 1)
                        # SFX: collect_bad.wav
        
        return reward

    def _check_termination(self):
        if self.game_over:
            return True
        if self.score >= self.WIN_SCORE:
            # SFX: win_jingle.wav
            return True
        if all(n['collected'] for n in self.nodes) and self.game_phase == "LAUNCHED":
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_slow_fuel": self.time_slow_fuel,
            "game_phase": self.game_phase
        }

    def _generate_level(self):
        # Update difficulty based on score
        self.difficulty_level = self.score // 200

        # Generate stars
        if not self.stars:
            self.stars = [
                (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.integers(1, 3))
                for _ in range(100)
            ]

        # Generate nodes and obstacles
        self.nodes.clear()
        self.obstacles.clear()
        
        num_nodes = self.np_random.integers(5, 9)
        num_obstacles = 2 + self.difficulty_level

        occupied_areas = []

        for _ in range(num_nodes):
            while True:
                pos = pygame.Vector2(
                    self.np_random.integers(50, self.WIDTH - 50),
                    self.np_random.integers(50, self.HEIGHT - 100)
                )
                new_area = pygame.Rect(pos.x - 30, pos.y - 30, 60, 60)
                if not any(new_area.colliderect(area) for area in occupied_areas):
                    self.nodes.append({
                        'pos': pos,
                        'color': self.COLOR_PALETTE[self.np_random.integers(0, len(self.COLOR_PALETTE))],
                        'collected': False
                    })
                    occupied_areas.append(new_area)
                    break
        
        for _ in range(num_obstacles):
            while True:
                size = self.np_random.integers(20, 40 + self.difficulty_level * 5)
                pos = pygame.Vector2(
                    self.np_random.integers(50, self.WIDTH - 50 - size),
                    self.np_random.integers(50, self.HEIGHT - 100 - size)
                )
                new_area = pygame.Rect(pos.x, pos.y, size, size)
                if not any(new_area.colliderect(area) for area in occupied_areas):
                    self.obstacles.append(new_area)
                    occupied_areas.append(new_area)
                    break

    def _render_game(self):
        # Render nodes and obstacles
        for obst in self.obstacles:
            pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, obst, border_radius=4)
        
        for node in self.nodes:
            if not node['collected']:
                self._draw_glow_circle(self.screen, node['color'], node['pos'], self.NODE_RADIUS, 15)

        # Render comet and its effects
        if self.game_phase == "AIMING":
            self._render_aim_indicator()
        elif self.game_phase == "LAUNCHED":
            self._render_trail()

        if not (self.game_phase == "LAUNCHED" and self.game_over):
            self._draw_glow_circle(self.screen, self.comet_color, self.comet_pos, self.COMET_RADIUS, 20)
        
        self._update_and_draw_particles()

        if self.time_slow_active:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_TIME_SLOW_OVERLAY)
            self.screen.blit(overlay, (0,0))
    
    def _render_background(self):
        for x, y, r in self.stars:
            c = self.np_random.integers(50, 100)
            pygame.gfxdraw.pixel(self.screen, x, y, (c, c, c+20))
            if r > 1:
                pygame.gfxdraw.pixel(self.screen, x+1, y, (c, c, c+20))
                pygame.gfxdraw.pixel(self.screen, x, y+1, (c, c, c+20))

    def _render_aim_indicator(self):
        angle_rad = math.radians(self.launch_angle)
        length = 30 + self.launch_power * 3
        end_pos = self.comet_pos + pygame.Vector2(math.cos(angle_rad), math.sin(angle_rad)) * length
        pygame.draw.line(self.screen, self.COLOR_WHITE, self.comet_pos, end_pos, 2)

    def _render_trail(self):
        if len(self.comet_trail) > 1:
            for i in range(len(self.comet_trail) - 1):
                alpha = int(255 * (i / len(self.comet_trail)))
                color = (*self.comet_color, alpha)
                start = self.comet_trail[i]
                end = self.comet_trail[i+1]
                
                temp_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(temp_surf, color, start, end, max(1, int(self.COMET_RADIUS * 0.5 * (i/len(self.comet_trail)))))
                self.screen.blit(temp_surf, (0,0))

    def _render_ui(self):
        # Score bar
        score_ratio = min(1.0, self.score / self.WIN_SCORE)
        bar_width = self.WIDTH * 0.4
        bar_height = 20
        bar_x = (self.WIDTH - bar_width) / 2
        
        pygame.draw.rect(self.screen, (50, 50, 70), (bar_x, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PALETTE[3], (bar_x, 10, bar_width * score_ratio, bar_height))
        score_text = self.font_small.render(f"ENERGY: {self.score}/{self.WIN_SCORE}", True, self.COLOR_WHITE)
        self.screen.blit(score_text, (bar_x + 5, 12))

        # Time slow fuel bar
        fuel_bar_width = 150
        pygame.draw.rect(self.screen, (40, 40, 40), (self.WIDTH - fuel_bar_width - 10, 10, fuel_bar_width, 10))
        fuel_ratio = self.time_slow_fuel / self.TIME_SLOW_MAX_FUEL
        pygame.draw.rect(self.screen, self.COLOR_PALETTE[2], (self.WIDTH - fuel_bar_width - 10, 10, fuel_bar_width * fuel_ratio, 10))
        
        if self.game_over:
            msg = "VICTORY!" if self.score >= self.WIN_SCORE else "GAME OVER"
            color = self.COLOR_PALETTE[1] if self.score >= self.WIN_SCORE else self.COLOR_PALETTE[0]
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _draw_glow_circle(self, surface, color, center, radius, glow_strength):
        center_int = (int(center.x), int(center.y))
        for i in range(glow_strength, 0, -2):
            alpha = 100 - (i / glow_strength * 100)
            glow_color = (*color, int(alpha))
            temp_surf = pygame.Surface((radius * 2 + i * 2, radius * 2 + i * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (radius + i, radius + i), radius + i)
            surface.blit(temp_surf, (center_int[0] - radius - i, center_int[1] - radius - i))
        pygame.gfxdraw.aacircle(surface, center_int[0], center_int[1], int(radius), color)
        pygame.gfxdraw.filled_circle(surface, center_int[0], center_int[1], int(radius), color)

    def _spawn_particles(self, pos, color, count, speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.np_random.uniform(1, speed)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': vel,
                'lifetime': self.np_random.integers(20, 40),
                'color': color
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['vel'] *= 0.95
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifetime'] / 40))
                color = (*p['color'], alpha)
                radius = int(3 * (p['lifetime'] / 40))
                if radius > 0:
                    temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
                    pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                    self.screen.blit(temp_surf, p['pos'] - pygame.Vector2(radius, radius))

    def _get_closest_dist(self, entity_type):
        min_dist = float('inf')
        if entity_type == 'node':
            entities = [n for n in self.nodes if not n['collected'] and tuple(n['color']) == tuple(self.comet_color)]
            if not entities: return min_dist
            for entity in entities:
                min_dist = min(min_dist, self.comet_pos.distance_to(entity['pos']))
        elif entity_type == 'obstacle':
            if not self.obstacles: return min_dist
            for entity in self.obstacles:
                min_dist = min(min_dist, self.comet_pos.distance_to(pygame.Vector2(entity.center)))
        return min_dist

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # The main script sets the video driver to dummy, but for manual play,
    # we need a real display. Pygame must be initialized again.
    pygame.quit()
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
    pygame.init()

    screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    pygame.display.set_caption("Comet Nebula")
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    while running:
        if terminated:
            # Wait for a moment on the game over screen, then reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

        # --- Manual Control Mapping ---
        keys = pygame.key.get_pressed()
        movement = 0 # None
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame Rendering ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)
        
    env.close()