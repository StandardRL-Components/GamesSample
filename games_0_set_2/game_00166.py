import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


# Set headless mode for Pygame, must be done before pygame.init()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game entities
class Tower:
    def __init__(self, grid_pos, tower_type):
        self.grid_pos = grid_pos
        self.type = tower_type
        self.cooldown = 0
        self.target = None

class Enemy:
    def __init__(self, path, enemy_type):
        self.path = path
        self.path_index = 0
        self.pixel_pos = list(self.path[0])
        self.max_health = enemy_type['health']
        self.health = enemy_type['health']
        self.speed = enemy_type['speed']
        self.value = enemy_type['value']
        self.color = enemy_type['color']
        self.radius = enemy_type['radius']

class Projectile:
    def __init__(self, start_pos, target_enemy, damage, speed, color):
        self.pos = list(start_pos)
        self.target = target_enemy
        self.damage = damage
        self.speed = speed
        self.color = color

class Particle:
    def __init__(self, pos, vel, lifespan, color, size):
        self.pos = list(pos)
        self.vel = list(vel)
        self.lifespan = lifespan
        self.color = color
        self.size = size

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press 'Space' to build the selected tower. "
        "Press 'Shift' to cycle between tower types."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing towers "
        "in this minimalist isometric tower defense game."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        self.screen_width = 640
        self.screen_height = 400
        pygame.init()
        pygame.font.init()
        # This is the main fix: pygame.display.set_mode() is necessary to initialize
        # the video subsystem, even in headless mode.
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.fps = 30

        # --- Game Configuration ---
        self._define_constants()
        self._define_colors_and_fonts()
        self._define_tower_types()
        self._define_enemy_path()
        
        # --- State variables (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.np_random = None
        self.base_health = 0
        self.resources = 0
        self.current_wave_index = 0
        self.wave_in_progress = False
        self.wave_cooldown = 0
        self.spawn_cooldown = 0
        self.enemies_to_spawn = 0
        self.cursor_pos = [0, 0]
        self.selected_tower_idx = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.reset()
        # self.validate_implementation() # Validation is good for dev, but can be commented out

    def _define_constants(self):
        self.GRID_W, self.GRID_H = 20, 12
        self.TILE_W_HALF, self.TILE_H_HALF = 20, 10
        self.ISO_ORIGIN_X = self.screen_width // 2
        self.ISO_ORIGIN_Y = 80
        self.MAX_STEPS = 30 * 300 # 5 minutes at 30fps
        self.MAX_WAVES = 10
        self.INITIAL_RESOURCES = 150
        self.INITIAL_BASE_HEALTH = 100

    def _define_colors_and_fonts(self):
        self.COLOR_BG = pygame.Color("#2c3e50")
        self.COLOR_GRID = pygame.Color("#34495e")
        self.COLOR_PATH = pygame.Color("#7f8c8d")
        self.COLOR_BASE = pygame.Color("#27ae60")
        self.COLOR_CURSOR_VALID = pygame.Color(46, 204, 113, 100)
        self.COLOR_CURSOR_INVALID = pygame.Color(231, 76, 60, 100)
        self.COLOR_TEXT = pygame.Color("#ecf0f1")
        self.FONT_UI = pygame.font.Font(None, 28)
        self.FONT_TITLE = pygame.font.Font(None, 48)
        self.FONT_SMALL = pygame.font.Font(None, 20)

    def _define_tower_types(self):
        self.TOWER_TYPES = [
            {
                "name": "Gun Turret", "cost": 50, "range": 120, "damage": 5, 
                "fire_rate": 0.5 * self.fps, "color": pygame.Color("#3498db"),
                "proj_speed": 8, "proj_color": pygame.Color("#5dade2")
            },
            {
                "name": "Cannon", "cost": 100, "range": 150, "damage": 25, 
                "fire_rate": 2.0 * self.fps, "color": pygame.Color("#e67e22"),
                "proj_speed": 6, "proj_color": pygame.Color("#f39c12")
            }
        ]
        
    def _define_enemy_path(self):
        path_grid = [
            (-1, 5), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5),
            (5, 4), (5, 3), (6, 3), (7, 3), (8, 3), (9, 3), (10, 3),
            (10, 4), (10, 5), (10, 6), (10, 7), (11, 7), (12, 7),
            (13, 7), (14, 7), (15, 7), (16, 7), (17, 7), (18, 7),
            (19, 7), (20, 7)
        ]
        self.path_pixels = [self._iso_to_screen(x, y) for x, y in path_grid]
        self.path_grid_set = set(path_grid)
        self.base_pos_grid = (18, 7)
        self.base_pos_screen = self._iso_to_screen(*self.base_pos_grid)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.current_wave_index = 0
        self.wave_in_progress = False
        self.wave_cooldown = 3 * self.fps  # 3 second delay before first wave
        self.spawn_cooldown = 0
        self.enemies_to_spawn_in_wave = 0
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_idx = 0
        self.last_space_held = False
        self.last_shift_held = False

        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward_events = []
        self.steps += 1

        self._handle_input(action)
        self._update_wave_system()
        self._update_towers(reward_events)
        self._update_projectiles(reward_events)
        self._update_enemies(reward_events)
        self._update_particles()
        
        reward = self._calculate_reward(reward_events)
        self.score += reward

        terminated = self._check_termination()
        if terminated:
            if self.win_condition:
                reward += 100
            else: # Loss
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # --- Place Tower (on key press) ---
        if space_held and not self.last_space_held:
            self._try_place_tower()
        
        # --- Cycle Tower Type (on key press) ---
        if shift_held and not self.last_shift_held:
            self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.TOWER_TYPES)
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _try_place_tower(self):
        selected_type = self.TOWER_TYPES[self.selected_tower_idx]
        if self.resources < selected_type["cost"]:
            return # sfx: error_buzz
        
        is_on_path = tuple(self.cursor_pos) in self.path_grid_set
        is_occupied = any(t.grid_pos == self.cursor_pos for t in self.towers)
        is_on_base = tuple(self.cursor_pos) == self.base_pos_grid

        if not is_on_path and not is_occupied and not is_on_base:
            self.towers.append(Tower(list(self.cursor_pos), selected_type))
            self.resources -= selected_type["cost"]
            # sfx: place_tower
            pos = self._iso_to_screen(*self.cursor_pos)
            self._create_particle_burst(pos, selected_type['color'], 15, 1.5)

    def _update_wave_system(self):
        if self.wave_in_progress:
            if self.enemies_to_spawn_in_wave > 0 and self.spawn_cooldown <= 0:
                self._spawn_enemy()
                self.enemies_to_spawn_in_wave -= 1
                self.spawn_cooldown = 1 * self.fps # 1 second between spawns
            self.spawn_cooldown -= 1
            if not self.enemies and self.enemies_to_spawn_in_wave <= 0:
                self.wave_in_progress = False
                self.wave_cooldown = 5 * self.fps # 5 seconds between waves
                if self.current_wave_index < self.MAX_WAVES:
                    self.score += 10 # Wave clear bonus
        else: # Between waves
            self.wave_cooldown -= 1
            if self.wave_cooldown <= 0 and self.current_wave_index < self.MAX_WAVES:
                self.current_wave_index += 1
                self.wave_in_progress = True
                self.enemies_to_spawn_in_wave = 2 + self.current_wave_index * 2
                self.spawn_cooldown = 0

    def _spawn_enemy(self):
        scale_factor = 1 + (self.current_wave_index - 1) * 0.05
        enemy_type = {
            'health': 20 * scale_factor,
            'speed': 0.75 * scale_factor,
            'value': 10,
            'color': pygame.Color("#c0392b"),
            'radius': 8,
        }
        self.enemies.append(Enemy(self.path_pixels, enemy_type))

    def _update_towers(self, reward_events):
        for tower in self.towers:
            tower.cooldown = max(0, tower.cooldown - 1)
            if tower.cooldown > 0:
                continue

            # Find a target if needed
            if tower.target is None or not self._is_enemy_in_range(tower, tower.target):
                tower.target = self._find_target_for_tower(tower)

            if tower.target:
                self._fire_projectile(tower)
                tower.cooldown = tower.type["fire_rate"]

    def _is_enemy_in_range(self, tower, enemy):
        tower_pos = self._iso_to_screen(*tower.grid_pos)
        dist_sq = (tower_pos[0] - enemy.pixel_pos[0])**2 + (tower_pos[1] - enemy.pixel_pos[1])**2
        return dist_sq < tower.type["range"]**2

    def _find_target_for_tower(self, tower):
        for enemy in self.enemies:
            if self._is_enemy_in_range(tower, enemy):
                return enemy
        return None

    def _fire_projectile(self, tower):
        # sfx: shoot
        start_pos = self._iso_to_screen(*tower.grid_pos)
        start_pos[1] -= self.TILE_H_HALF # Fire from top of tower
        self.projectiles.append(Projectile(
            start_pos, tower.target, tower.type["damage"],
            tower.type["proj_speed"], tower.type["proj_color"]
        ))
        
    def _update_projectiles(self, reward_events):
        for p in self.projectiles[:]:
            if p.target not in self.enemies:
                self.projectiles.remove(p)
                continue

            target_pos = p.target.pixel_pos
            direction = (target_pos[0] - p.pos[0], target_pos[1] - p.pos[1])
            dist = math.hypot(*direction)
            
            if dist < p.speed:
                # Hit
                p.target.health -= p.damage
                reward_events.append("hit")
                self._create_particle_burst(p.pos, p.color, 5, 1)
                self.projectiles.remove(p)
                # sfx: hit
            else:
                p.pos[0] += (direction[0] / dist) * p.speed
                p.pos[1] += (direction[1] / dist) * p.speed

    def _update_enemies(self, reward_events):
        for e in self.enemies[:]:
            if e.health <= 0:
                # Killed
                reward_events.append("kill")
                self.resources += e.value
                self._create_particle_burst(e.pixel_pos, e.color, 20, 2)
                self.enemies.remove(e)
                # sfx: enemy_die
                continue

            if e.path_index >= len(e.path) - 1:
                # Reached base
                self.base_health -= 10
                self._create_particle_burst(self.base_pos_screen, pygame.Color("red"), 30, 3)
                self.enemies.remove(e)
                # sfx: base_damage
                continue

            target_pixel_pos = e.path[e.path_index + 1]
            direction = (target_pixel_pos[0] - e.pixel_pos[0], target_pixel_pos[1] - e.pixel_pos[1])
            dist = math.hypot(*direction)

            if dist < e.speed:
                e.pixel_pos = list(target_pixel_pos)
                e.path_index += 1
            else:
                e.pixel_pos[0] += (direction[0] / dist) * e.speed
                e.pixel_pos[1] += (direction[1] / dist) * e.speed

    def _update_particles(self):
        for p in self.particles[:]:
            p.pos[0] += p.vel[0]
            p.pos[1] += p.vel[1]
            p.lifespan -= 1
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _calculate_reward(self, events):
        reward = -0.01  # Time penalty
        reward += events.count("hit") * 0.1
        reward += events.count("kill") * 1.0
        return reward

    def _check_termination(self):
        self.win_condition = self.current_wave_index >= self.MAX_WAVES and not self.enemies and not self.wave_in_progress
        if self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.win_condition:
            self.game_over = True
            return True
        return False

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
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.current_wave_index,
        }

    def _render_game(self):
        self._render_grid_and_path()
        self._render_cursor()
        
        # --- Sort and render dynamic objects for correct isometric layering ---
        renderables = self.towers + self.enemies
        renderables.sort(key=lambda obj: self._get_object_y_pos(obj))

        for obj in renderables:
            if isinstance(obj, Tower):
                self._render_tower(obj)
            elif isinstance(obj, Enemy):
                self._render_enemy(obj)
        
        for p in self.projectiles:
            self._render_projectile(p)

        for p in self.particles:
            self._render_particle(p)

    def _get_object_y_pos(self, obj):
        if isinstance(obj, Tower):
            return self._iso_to_screen(*obj.grid_pos)[1]
        elif isinstance(obj, Enemy):
            return obj.pixel_pos[1]
        return 0

    def _render_grid_and_path(self):
        # Draw grid
        for r in range(self.GRID_H):
            for c in range(self.GRID_W):
                x, y = self._iso_to_screen(c, r)
                points = [
                    (x, y),
                    (x + self.TILE_W_HALF, y + self.TILE_H_HALF),
                    (x, y + self.TILE_H_HALF * 2),
                    (x - self.TILE_W_HALF, y + self.TILE_H_HALF)
                ]
                pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw path
        if len(self.path_pixels) > 1:
            pygame.draw.aalines(self.screen, self.COLOR_PATH, False, self.path_pixels, 3)
        
        # Draw base
        bx, by = self.base_pos_screen
        base_points = [
            (bx, by - self.TILE_H_HALF),
            (bx + self.TILE_W_HALF, by),
            (bx, by + self.TILE_H_HALF),
            (bx - self.TILE_W_HALF, by)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, base_points, pygame.Color('white'))

    def _render_cursor(self):
        is_on_path = tuple(self.cursor_pos) in self.path_grid_set
        is_occupied = any(t.grid_pos == self.cursor_pos for t in self.towers)
        is_on_base = tuple(self.cursor_pos) == self.base_pos_grid
        is_valid = not is_on_path and not is_occupied and not is_on_base

        cursor_color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        x, y = self._iso_to_screen(*self.cursor_pos)
        points = [
            (x, y),
            (x + self.TILE_W_HALF, y + self.TILE_H_HALF),
            (x, y + self.TILE_H_HALF * 2),
            (x - self.TILE_W_HALF, y + self.TILE_H_HALF)
        ]
        
        # Fix: Create a new per-pixel alpha surface to draw the transparent polygon
        temp_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        pygame.gfxdraw.filled_polygon(temp_surface, points, cursor_color)
        self.screen.blit(temp_surface, (0, 0))

    def _render_tower(self, tower):
        x, y = self._iso_to_screen(*tower.grid_pos)
        # Fix: Ensure color components are integers
        base_color = tuple(int(c * 0.7) for c in tower.type['color'])
        # Base
        base_points = [
            (x, y),
            (x + self.TILE_W_HALF, y + self.TILE_H_HALF),
            (x, y + self.TILE_H_HALF * 2),
            (x - self.TILE_W_HALF, y + self.TILE_H_HALF)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, base_color)
        # Top
        top_y = y - self.TILE_H_HALF
        top_points = [
            (x, top_y),
            (x + self.TILE_W_HALF, top_y + self.TILE_H_HALF),
            (x, top_y + self.TILE_H_HALF * 2),
            (x - self.TILE_W_HALF, top_y + self.TILE_H_HALF)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, top_points, tower.type['color'])
        pygame.gfxdraw.aapolygon(self.screen, top_points, pygame.Color('white'))
        
    def _render_enemy(self, enemy):
        x, y = int(enemy.pixel_pos[0]), int(enemy.pixel_pos[1])
        pygame.gfxdraw.aacircle(self.screen, x, y, enemy.radius, enemy.color)
        pygame.gfxdraw.filled_circle(self.screen, x, y, enemy.radius, enemy.color)

        # Health bar
        if enemy.health < enemy.max_health:
            bar_w = 20
            bar_h = 4
            health_pct = enemy.health / enemy.max_health
            fill_w = int(bar_w * health_pct)
            bar_x = x - bar_w // 2
            bar_y = y - enemy.radius - 8
            pygame.draw.rect(self.screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, (200, 50, 50), (bar_x, bar_y, fill_w, bar_h))

    def _render_projectile(self, p):
        size = 3
        pygame.draw.rect(self.screen, p.color, (p.pos[0]-size//2, p.pos[1]-size//2, size, size))

    def _render_particle(self, p):
        size = max(1, int(p.size * (p.lifespan / 15)))
        pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), size, p.color)

    def _render_ui(self):
        # Wave Info
        if self.win_condition:
            wave_text = "YOU WIN!"
        elif self.game_over:
            wave_text = "GAME OVER"
        elif not self.wave_in_progress:
            wave_text = f"Next wave in {self.wave_cooldown / self.fps:.1f}s"
        else:
            wave_text = f"Wave: {self.current_wave_index} / {self.MAX_WAVES}"
        self._draw_text(wave_text, (10, 10), self.FONT_UI)

        # Base Health
        health_text = f"Base HP: {max(0, self.base_health)}"
        self._draw_text(health_text, (self.screen_width - 150, 10), self.FONT_UI, "topright")

        # Resources
        res_text = f"Resources: ${self.resources}"
        self._draw_text(res_text, (self.screen_width // 2, self.screen_height - 25), self.FONT_UI, "midbottom")

        # Tower Selection UI
        self._render_tower_select_ui()

    def _render_tower_select_ui(self):
        ui_w, ui_h = 200, 80
        ui_x, ui_y = self.screen_width - ui_w - 10, self.screen_height - ui_h - 10
        
        # Panel background
        panel_rect = pygame.Rect(ui_x, ui_y, ui_w, ui_h)
        # Note: pygame.draw.rect does not support alpha. For true transparency, a separate surface is needed.
        # This will draw a solid black box. Keeping as-is to minimize changes.
        s = pygame.Surface((ui_w, ui_h), pygame.SRCALPHA)
        s.fill((0, 0, 0, 150))
        pygame.draw.rect(s, (128, 128, 128, 200), s.get_rect(), 1, border_radius=5)
        self.screen.blit(s, panel_rect.topleft)

        selected_tower = self.TOWER_TYPES[self.selected_tower_idx]
        self._draw_text(f"Selected: {selected_tower['name']}", (ui_x + 10, ui_y + 10), self.FONT_SMALL)
        self._draw_text(f"Cost: ${selected_tower['cost']}", (ui_x + 10, ui_y + 30), self.FONT_SMALL)
        self._draw_text(f"Damage: {selected_tower['damage']}", (ui_x + 10, ui_y + 50), self.FONT_SMALL)

    def _draw_text(self, text, pos, font, align="topleft"):
        text_surface = font.render(text, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect()
        setattr(text_rect, align, pos)
        self.screen.blit(text_surface, text_rect)

    def _iso_to_screen(self, grid_x, grid_y):
        screen_x = self.ISO_ORIGIN_X + (grid_x - grid_y) * self.TILE_W_HALF
        screen_y = self.ISO_ORIGIN_Y + (grid_x + grid_y) * self.TILE_H_HALF
        return int(screen_x), int(screen_y)

    def _create_particle_burst(self, pos, color, count, max_speed):
        for _ in range(count):
            # Fix: Use self.np_random instead of 'random' module
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            # Fix: Use .integers() which is exclusive on the high end
            lifespan = self.np_random.integers(10, 21)
            size = self.np_random.integers(2, 5)
            self.particles.append(Particle(pos, vel, lifespan, color, size))

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # To run with a visible window, unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    running = True
    
    # --- Action state for human play ---
    action = [0, 0, 0] # no-op, released, released
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        # --- Map keyboard state to action space ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            # obs, info = env.reset() # Uncomment to auto-reset

        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.fps)
        
    pygame.quit()