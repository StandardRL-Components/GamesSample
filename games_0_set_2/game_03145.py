
# Generated: 2025-08-28T07:08:01.326024
# Source Brief: brief_03145.md
# Brief Index: 3145

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the placement cursor. Press Shift to cycle tower types. "
        "Press Space to place a tower. Waves start automatically."
    )

    game_description = (
        "Defend your base from waves of zombies by strategically placing defensive towers "
        "in an isometric 2D environment. Survive all 20 waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    FPS = 30
    GRID_WIDTH, GRID_HEIGHT = 20, 20
    TILE_W, TILE_H = 24, 12
    ORIGIN_X, ORIGIN_Y = SCREEN_WIDTH // 2, 80

    # Colors
    COLOR_BG = (20, 25, 30)
    COLOR_GRID = (40, 50, 60)
    COLOR_PATH = (70, 60, 50)
    COLOR_PATH_BORDER = (90, 80, 70)
    COLOR_BASE = (0, 150, 50)
    COLOR_BASE_BORDER = (50, 200, 100)
    COLOR_ZOMBIE = (200, 50, 50)
    COLOR_ZOMBIE_BORDER = (255, 100, 100)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 255, 0, 150)
    COLOR_HEALTH_BG = (80, 0, 0)
    COLOR_HEALTH_FG = (0, 200, 0)
    COLOR_UI_BG = (10, 15, 20, 200)

    # Game Parameters
    BASE_MAX_HEALTH = 100
    ZOMBIE_DAMAGE = 10
    MAX_WAVES = 20
    MAX_STEPS = 15000 # ~8 minutes
    PLACEMENT_PHASE_SECONDS = 5
    MAX_TOWERS = 15

    TOWER_SPECS = {
        'MACHINE_GUN': {'range': 4.5, 'damage': 8, 'fire_rate': 6, 'color': (0, 180, 255), 'proj_speed': 8, 'proj_size': 3},
        'CANNON': {'range': 6.5, 'damage': 40, 'fire_rate': 45, 'color': (255, 150, 0), 'proj_speed': 6, 'proj_size': 5},
    }

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
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        self.path_grid_coords = self._define_path()
        self.path_pixels = [self._to_screen_coords(p[0], p[1]) for p in self.path_grid_coords]
        self.base_pos_grid = self.path_grid_coords[-1]
        
        self.tower_types = list(self.TOWER_SPECS.keys())
        self.placement_grid = self._create_placement_grid()
        
        self.np_random = None
        self.game_over = False
        self.steps = 0
        self.score = 0

        self.reset()
        
        # This is a good place for the validation check as it needs a reset state
        self.validate_implementation()

    def _define_path(self):
        return [
            (-1, 8), (2, 8), (2, 3), (7, 3), (7, 12),
            (12, 12), (12, 7), (17, 7), (17, 10), (20, 10)
        ]

    def _create_placement_grid(self):
        grid = set()
        path_set = set(self.path_grid_coords)
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if (c, r) not in path_set and (c-1, r) not in path_set and (c+1, r) not in path_set and (c, r-1) not in path_set and (c, r+1) not in path_set:
                     grid.add((c, r))
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.BASE_MAX_HEALTH
        self.wave_number = 0
        
        self.zombies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.zombies_to_spawn = 0
        self.zombie_spawn_cooldown = 0
        
        self.placement_cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_tower_idx = 0
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self._start_placement_phase()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1

        self._handle_input(action)
        
        if self.game_phase == 'placement':
            self.placement_timer -= 1
            if self.placement_timer <= 0:
                self._start_wave()
        
        elif self.game_phase == 'wave':
            wave_reward = self._update_wave()
            reward += wave_reward

            if not self.zombies and self.zombies_to_spawn == 0:
                reward += 1.0 # Wave complete bonus
                self.score += 1
                if self.wave_number >= self.MAX_WAVES:
                    self.game_over = True
                    reward += 100.0
                else:
                    self._start_placement_phase()

        self._update_projectiles_and_particles()

        if self.base_health <= 0:
            self.base_health = 0
            self.game_over = True
            reward -= 100.0

        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.game_phase == 'placement':
            # Move cursor
            if movement == 1: self.placement_cursor_pos = (self.placement_cursor_pos[0], max(0, self.placement_cursor_pos[1] - 1))
            elif movement == 2: self.placement_cursor_pos = (self.placement_cursor_pos[0], min(self.GRID_HEIGHT - 1, self.placement_cursor_pos[1] + 1))
            elif movement == 3: self.placement_cursor_pos = (max(0, self.placement_cursor_pos[0] - 1), self.placement_cursor_pos[1])
            elif movement == 4: self.placement_cursor_pos = (min(self.GRID_WIDTH - 1, self.placement_cursor_pos[0] + 1), self.placement_cursor_pos[1])

            # Cycle tower type
            if shift_held and not self.last_shift_held:
                self.selected_tower_idx = (self.selected_tower_idx + 1) % len(self.tower_types)
                # sfx: UI_cycle_sound

            # Place tower
            if space_held and not self.last_space_held:
                if self.placement_cursor_pos in self.placement_grid and len(self.towers) < self.MAX_TOWERS:
                    is_occupied = any(t['grid_pos'] == self.placement_cursor_pos for t in self.towers)
                    if not is_occupied:
                        spec_name = self.tower_types[self.selected_tower_idx]
                        spec = self.TOWER_SPECS[spec_name]
                        screen_pos = self._to_screen_coords(*self.placement_cursor_pos)
                        new_tower = {
                            'grid_pos': self.placement_cursor_pos,
                            'pos': pygame.math.Vector2(screen_pos),
                            'type': spec_name,
                            'range_sq': (spec['range'] * self.TILE_W) ** 2,
                            'cooldown': 0,
                        }
                        self.towers.append(new_tower)
                        # sfx: tower_place_sound

        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _start_placement_phase(self):
        self.game_phase = 'placement'
        self.placement_timer = self.PLACEMENT_PHASE_SECONDS * self.FPS

    def _start_wave(self):
        self.game_phase = 'wave'
        self.wave_number += 1
        
        self.zombies_to_spawn = 5 + (self.wave_number - 1) * 2
        self.zombie_spawn_cooldown = 0

    def _update_wave(self):
        reward = 0
        # Spawn new zombies
        self.zombie_spawn_cooldown -= 1
        if self.zombies_to_spawn > 0 and self.zombie_spawn_cooldown <= 0:
            self._spawn_zombie()
            self.zombies_to_spawn -= 1
            self.zombie_spawn_cooldown = self.FPS // (1 + self.wave_number / 5) # Spawn faster in later waves

        # Update towers
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    spec = self.TOWER_SPECS[tower['type']]
                    tower['cooldown'] = spec['fire_rate']
                    self._create_projectile(tower, target)
                    # sfx: tower_fire_sound

        # Update zombies
        for z in self.zombies[:]:
            dist_to_next_waypoint = z['pos'].distance_to(z['target_pixel'])
            if dist_to_next_waypoint < z['speed']:
                z['waypoint_idx'] += 1
                if z['waypoint_idx'] >= len(self.path_pixels):
                    self.base_health -= self.ZOMBIE_DAMAGE
                    reward -= 0.1 * self.ZOMBIE_DAMAGE
                    self.score -= 0.1 * self.ZOMBIE_DAMAGE
                    self.zombies.remove(z)
                    self._create_particles(z['pos'], self.COLOR_BASE, 20)
                    # sfx: base_damage_sound
                    continue
                z['target_pixel'] = pygame.math.Vector2(self.path_pixels[z['waypoint_idx']])
            
            direction = (z['target_pixel'] - z['pos']).normalize()
            z['pos'] += direction * z['speed']
        
        return reward

    def _spawn_zombie(self):
        health = 100 * (1.05 ** (self.wave_number - 1))
        speed = 1.0 * (1.02 ** (self.wave_number - 1))
        
        start_pos = pygame.math.Vector2(self.path_pixels[0])
        # Add small random offset to start position
        start_pos += (self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5))

        new_zombie = {
            'pos': start_pos,
            'health': health,
            'max_health': health,
            'speed': speed,
            'waypoint_idx': 1,
            'target_pixel': pygame.math.Vector2(self.path_pixels[1]),
        }
        self.zombies.append(new_zombie)

    def _find_target(self, tower):
        for z in self.zombies:
            if tower['pos'].distance_squared_to(z['pos']) < tower['range_sq']:
                return z
        return None

    def _create_projectile(self, tower, target):
        spec = self.TOWER_SPECS[tower['type']]
        proj = {
            'pos': tower['pos'].copy(),
            'target': target,
            'speed': spec['proj_speed'],
            'damage': spec['damage'],
            'color': spec['color'],
            'size': spec['proj_size']
        }
        self.projectiles.append(proj)

    def _update_projectiles_and_particles(self):
        # Projectiles
        for p in self.projectiles[:]:
            if p['target'] not in self.zombies: # Target already dead
                self.projectiles.remove(p)
                continue
            
            direction = (p['target']['pos'] - p['pos'])
            dist = direction.length()
            
            if dist < p['speed']:
                p['target']['health'] -= p['damage']
                self._create_particles(p['pos'], p['color'], 5)
                # sfx: projectile_hit_sound
                if p['target']['health'] <= 0:
                    self._create_particles(p['target']['pos'], self.COLOR_ZOMBIE, 15)
                    self.zombies.remove(p['target'])
                    self.score += 0.1 # Kill reward
                    # sfx: zombie_death_sound
                self.projectiles.remove(p)
            else:
                p['pos'] += direction.normalize() * p['speed']

        # Particles
        for part in self.particles[:]:
            part['pos'] += part['vel']
            part['lifespan'] -= 1
            if part['lifespan'] <= 0:
                self.particles.remove(part)

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = pygame.math.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({'pos': pos.copy(), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_and_path()
        
        # Combine and sort all drawable entities for correct isometric layering
        drawables = []
        for t in self.towers:
            drawables.append(('tower', t))
        for z in self.zombies:
            drawables.append(('zombie', z))

        drawables.sort(key=lambda item: item[1]['pos'].y)

        for item_type, item in drawables:
            if item_type == 'tower':
                self._render_tower(item)
            elif item_type == 'zombie':
                self._render_zombie(item)

        if self.game_phase == 'placement':
            self._render_placement_cursor()

        for p in self.projectiles:
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'].x), int(p['pos'].y)), p['size'])

        for part in self.particles:
            size = max(1, int(part['lifespan'] / 4))
            pygame.draw.circle(self.screen, part['color'], (int(part['pos'].x), int(part['pos'].y)), size)

    def _render_grid_and_path(self):
        # Draw grid
        for r in range(self.GRID_HEIGHT + 1):
            p1 = self._to_screen_coords(0, r)
            p2 = self._to_screen_coords(self.GRID_WIDTH, r)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.GRID_WIDTH + 1):
            p1 = self._to_screen_coords(c, 0)
            p2 = self._to_screen_coords(c, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        
        # Draw path
        for i in range(len(self.path_pixels) - 1):
            pygame.draw.line(self.screen, self.COLOR_PATH_BORDER, self.path_pixels[i], self.path_pixels[i+1], self.TILE_H * 2 + 4)
            pygame.draw.line(self.screen, self.COLOR_PATH, self.path_pixels[i], self.path_pixels[i+1], self.TILE_H * 2)

        # Draw Base
        base_screen_pos = self._to_screen_coords(*self.base_pos_grid)
        base_points = self._get_iso_poly(base_screen_pos[0], base_screen_pos[1])
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, base_points, self.COLOR_BASE_BORDER)

    def _render_tower(self, tower):
        spec = self.TOWER_SPECS[tower['type']]
        pos = tower['pos']
        
        # Base
        base_points = self._get_iso_poly(pos.x, pos.y, scale=0.8)
        pygame.gfxdraw.filled_polygon(self.screen, base_points, (30,30,30))
        pygame.gfxdraw.aapolygon(self.screen, base_points, (60,60,60))
        
        # Turret
        if spec['damage'] > 20: # Cannon
            pygame.draw.circle(self.screen, spec['color'], (int(pos.x), int(pos.y - self.TILE_H*0.75)), 6)
        else: # Machine Gun
            pygame.draw.rect(self.screen, spec['color'], (int(pos.x-4), int(pos.y - self.TILE_H*0.75 - 4), 8, 8))


    def _render_zombie(self, zombie):
        pos = zombie['pos']
        bob = math.sin(self.steps * 0.2 + pos.x) * 2
        
        points = self._get_iso_poly(pos.x, pos.y + bob, scale=0.6)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_ZOMBIE)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ZOMBIE_BORDER)

        # Health bar
        bar_width = 20
        bar_height = 4
        health_pct = zombie['health'] / zombie['max_health']
        bar_x = pos.x - bar_width // 2
        bar_y = pos.y - self.TILE_H - 10 + bob
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, int(bar_width * health_pct), bar_height))

    def _render_placement_cursor(self):
        grid_x, grid_y = self.placement_cursor_pos
        screen_x, screen_y = self._to_screen_coords(grid_x, grid_y)
        
        is_valid = self.placement_cursor_pos in self.placement_grid and \
                   len(self.towers) < self.MAX_TOWERS and \
                   not any(t['grid_pos'] == self.placement_cursor_pos for t in self.towers)

        color = (0, 255, 0, 100) if is_valid else (255, 0, 0, 100)
        
        # Draw cursor
        points = self._get_iso_poly(screen_x, screen_y)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Draw range indicator
        spec_name = self.tower_types[self.selected_tower_idx]
        spec = self.TOWER_SPECS[spec_name]
        radius = int(spec['range'] * self.TILE_W)
        pygame.gfxdraw.aacircle(self.screen, int(screen_x), int(screen_y), radius, (255, 255, 255, 50))


    def _render_ui(self):
        # Info Panel
        panel_rect = pygame.Rect(5, 5, 180, 85)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, panel_rect, border_radius=5)
        
        # Base Health
        health_text = self.font_medium.render(f"Base Health", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (15, 12))
        health_pct = self.base_health / self.BASE_MAX_HEALTH
        health_bar_rect = pygame.Rect(15, 35, 160, 12)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, health_bar_rect)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (15, 35, int(160 * health_pct), 12))

        # Wave
        wave_text = self.font_medium.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (15, 55))

        # Score
        score_text = self.font_small.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 75))

        # Placement Info
        if self.game_phase == 'placement':
            time_left = self.placement_timer / self.FPS
            placement_text = self.font_large.render(f"PLAN YOUR DEFENSE: {time_left:.1f}", True, self.COLOR_TEXT)
            text_rect = placement_text.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
            self.screen.blit(placement_text, text_rect)

            # Selected tower
            spec_name = self.tower_types[self.selected_tower_idx]
            tower_text = self.font_medium.render(f"Selected: {spec_name}", True, self.COLOR_TEXT)
            self.screen.blit(tower_text, (self.SCREEN_WIDTH - 180, 15))
            towers_placed_text = self.font_small.render(f"Towers: {len(self.towers)}/{self.MAX_TOWERS}", True, self.COLOR_TEXT)
            self.screen.blit(towers_placed_text, (self.SCREEN_WIDTH - 180, 45))

    def _to_screen_coords(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * self.TILE_W
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * self.TILE_H
        return screen_x, screen_y

    def _get_iso_poly(self, screen_x, screen_y, scale=1.0):
        w = self.TILE_W * scale
        h = self.TILE_H * scale
        return [
            (int(screen_x), int(screen_y - h)),
            (int(screen_x + w), int(screen_y)),
            (int(screen_x), int(screen_y + h)),
            (int(screen_x - w), int(screen_y)),
        ]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
            "zombies": len(self.zombies),
            "towers": len(self.towers)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")