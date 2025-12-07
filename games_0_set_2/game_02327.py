
# Generated: 2025-08-28T04:28:17.691297
# Source Brief: brief_02327.md
# Brief Index: 2327

        
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
        "Controls: ↑↓←→ to select a build spot. Hold Shift to cycle tower type. Press Space to build."
    )

    game_description = (
        "Defend your base from waves of geometric enemies by strategically placing towers along their path."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_PATH = (45, 50, 66)
    COLOR_PLACE_TILE = (60, 66, 82)
    COLOR_PLACE_TILE_HOVER = (80, 88, 110)
    COLOR_BASE = (66, 165, 245)
    COLOR_TEXT = (220, 220, 220)
    COLOR_SCORE = (255, 202, 40)
    COLOR_HEALTH_BAR_BG = (80, 80, 80)
    COLOR_HEALTH_BAR_PLAYER = (76, 175, 80)
    COLOR_HEALTH_BAR_ENEMY = (239, 83, 80)
    COLOR_FLASH_DAMAGE = (200, 0, 0, 100)

    TOWER_COLORS = {
        'blue': (64, 196, 255),
        'yellow': (255, 235, 59),
        'purple': (186, 104, 200),
    }

    # Screen & Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 32, 20
    CELL_SIZE = 20

    # Game Mechanics
    MAX_STEPS = 15000 # ~8 minutes at 30fps
    TOTAL_WAVES = 10
    BASE_START_HEALTH = 100
    STARTING_MONEY = 150
    WAVE_PREP_TIME = 150 # 5 seconds

    # --- Initialization ---
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("sans", 20)
        self.font_wave = pygame.font.SysFont("sans", 24, bold=True)

        # Game Data
        self._define_game_data()

        # State variables (initialized in reset)
        self.steps = 0
        self.score = 0
        self.money = 0
        self.game_over = False
        self.base_health = 0
        self.wave_index = 0
        self.wave_spawner = {}
        self.wave_timer = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_index = 0
        self.selected_tower_type = 'blue'
        self.last_action = [0, 0, 0]
        self.screen_flash_alpha = 0

        self.reset()
        self.validate_implementation()

    def _define_game_data(self):
        """Defines static data like paths, tower specs, etc."""
        # Path waypoints in grid coordinates
        self.path_waypoints_grid = [
            (-1, 9), (4, 9), (4, 4), (12, 4), (12, 15),
            (22, 15), (22, 9), (27, 9), (27, 2), (32, 2)
        ]
        self.path_waypoints_px = [( (p[0] + 0.5) * self.CELL_SIZE, (p[1] + 0.5) * self.CELL_SIZE ) for p in self.path_waypoints_grid]

        # Tower placement spots in grid coordinates
        spots = [
            (2, 6), (2, 7), (2, 8), (6, 6), (6, 5), (6, 4), (6, 3), (10, 6), (10, 5),
            (14, 6), (14, 7), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 13),
            (20, 13), (20, 12), (20, 11), (20, 10), (24, 11), (24, 12), (24, 13), (24, 7),
            (24, 6), (24, 5), (24, 4), (29, 4), (29, 5), (29, 6), (29, 7)
        ]
        self.placement_spots = sorted(spots, key=lambda p: (p[1], p[0])) # Sort by row, then col

        # Tower specifications
        self.TOWER_SPECS = {
            'blue': {'cost': 25, 'damage': 5, 'range': 90, 'fire_rate': 15, 'proj_speed': 8, 'aoe': 0},
            'yellow': {'cost': 40, 'damage': 20, 'range': 120, 'fire_rate': 45, 'proj_speed': 10, 'aoe': 0},
            'purple': {'cost': 60, 'damage': 12, 'range': 100, 'fire_rate': 60, 'proj_speed': 6, 'aoe': 40},
        }
        self.TOWER_TYPES = list(self.TOWER_SPECS.keys())

        # Wave definitions
        self.WAVES = []
        for i in range(self.TOTAL_WAVES):
            self.WAVES.append({
                'count': 5 + i * 2,
                'health': 10 + i * 5,
                'speed': 0.8 + i * 0.1,
                'spawn_delay': max(10, 30 - i * 2),
                'money_reward': 2 + i // 2
            })

    # --- Gymnasium Core Methods ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.score = 0
        self.money = self.STARTING_MONEY
        self.game_over = False
        self.base_health = self.BASE_START_HEALTH
        self.wave_index = -1
        self.wave_timer = self.WAVE_PREP_TIME
        self.wave_spawner = {}
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_index = 0
        self.selected_tower_type = 'blue'
        self.last_action = [0, 0, 0]
        self.screen_flash_alpha = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.001  # Small penalty for time passing
        self.steps += 1

        # Handle wave progression
        self._update_wave_state()

        # Handle player input (only if not in pre-wave countdown)
        if not (self.wave_timer > 0 and not self.enemies and self.wave_index > -1):
            input_cost = self._handle_input(action)
            reward -= input_cost / 100 # Penalize spending money slightly

        # Update game state
        self._update_towers()
        reward += self._update_projectiles()
        enemy_update = self._update_enemies()
        reward += enemy_update['reward']
        if enemy_update['base_damage'] > 0:
            self.screen_flash_alpha = 200 # Trigger screen flash

        self._update_particles()
        self._update_effects()

        # Update last action state
        self.last_action = action

        # Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.base_health <= 0 or self.steps >= self.MAX_STEPS:
                reward -= 100 # Loss
            elif self.wave_index >= self.TOTAL_WAVES:
                reward += 100 # Win
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- State Update Helpers ---
    def _handle_input(self, action):
        movement, space_btn, shift_btn = action[0], action[1], action[2]
        
        # --- Action debouncing (detect press, not hold) ---
        move_pressed = movement != 0 and self.last_action[0] == 0
        space_pressed = space_btn == 1 and self.last_action[1] == 0
        shift_pressed = shift_btn == 1 and self.last_action[2] == 0

        # Cursor movement
        if move_pressed:
            self._move_cursor(movement)

        # Cycle tower type
        if shift_pressed:
            current_idx = self.TOWER_TYPES.index(self.selected_tower_type)
            self.selected_tower_type = self.TOWER_TYPES[(current_idx + 1) % len(self.TOWER_TYPES)]
            # sfx: ui_chime

        # Place tower
        if space_pressed:
            return self._place_tower()
        
        return 0

    def _move_cursor(self, direction):
        if not self.placement_spots: return
        
        curr_x, curr_y = self.placement_spots[self.cursor_index]
        
        # Find all spots in the same row or column
        row_spots = sorted([i for i, (x, y) in enumerate(self.placement_spots) if y == curr_y])
        col_spots = sorted([i for i, (x, y) in enumerate(self.placement_spots) if x == curr_x])

        def find_next(current_list, current_val, step):
            try:
                idx = current_list.index(current_val)
                return current_list[(idx + step) % len(current_list)]
            except (ValueError, IndexError):
                return current_val

        if direction == 1: # Up
            self.cursor_index = find_next(col_spots, self.cursor_index, -1)
        elif direction == 2: # Down
            self.cursor_index = find_next(col_spots, self.cursor_index, 1)
        elif direction == 3: # Left
            self.cursor_index = find_next(row_spots, self.cursor_index, -1)
        elif direction == 4: # Right
            self.cursor_index = find_next(row_spots, self.cursor_index, 1)
        # sfx: ui_tick

    def _place_tower(self):
        spec = self.TOWER_SPECS[self.selected_tower_type]
        if self.money < spec['cost']:
            # sfx: ui_error
            return 0
        
        grid_pos = self.placement_spots[self.cursor_index]
        is_occupied = any(t['grid_pos'] == grid_pos for t in self.towers)

        if not is_occupied:
            self.money -= spec['cost']
            px_pos = ((grid_pos[0] + 0.5) * self.CELL_SIZE, (grid_pos[1] + 0.5) * self.CELL_SIZE)
            self.towers.append({
                'grid_pos': grid_pos,
                'px_pos': px_pos,
                'type': self.selected_tower_type,
                'spec': spec,
                'cooldown': 0,
                'target': None
            })
            # sfx: build_tower
            return spec['cost']
        else:
            # sfx: ui_error
            return 0

    def _update_wave_state(self):
        all_spawned = self.wave_spawner.get('count', 0) <= 0
        if not self.enemies and all_spawned and self.wave_index < self.TOTAL_WAVES:
            self.wave_timer -= 1
            if self.wave_timer <= 0:
                self._start_next_wave()
        
        if self.wave_spawner.get('count', 0) > 0:
            self.wave_spawner['timer'] -= 1
            if self.wave_spawner['timer'] <= 0:
                self._spawn_enemy()
                self.wave_spawner['timer'] = self.wave_spawner['delay']

    def _start_next_wave(self):
        self.wave_index += 1
        if self.wave_index >= self.TOTAL_WAVES:
            return
        
        wave_data = self.WAVES[self.wave_index]
        self.wave_spawner = {
            'count': wave_data['count'],
            'health': wave_data['health'],
            'speed': wave_data['speed'],
            'delay': wave_data['spawn_delay'],
            'timer': 0,
            'money_reward': wave_data['money_reward']
        }
        self.wave_timer = self.WAVE_PREP_TIME

    def _spawn_enemy(self):
        self.enemies.append({
            'pos': np.array(self.path_waypoints_px[0], dtype=float),
            'health': self.wave_spawner['health'],
            'max_health': self.wave_spawner['health'],
            'speed': self.wave_spawner['speed'],
            'path_target_idx': 1,
            'id': self.np_random.integers(1, 1e9),
            'money_reward': self.wave_spawner['money_reward']
        })
        self.wave_spawner['count'] -= 1
        # sfx: enemy_spawn

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue

            # Find target
            tower['target'] = None
            min_dist = tower['spec']['range']
            for enemy in self.enemies:
                dist = np.linalg.norm(np.array(tower['px_pos']) - enemy['pos'])
                if dist < min_dist:
                    min_dist = dist
                    tower['target'] = enemy['id']
            
            # Fire projectile
            if tower['target'] is not None:
                tower['cooldown'] = tower['spec']['fire_rate']
                self.projectiles.append({
                    'pos': np.array(tower['px_pos'], dtype=float),
                    'spec': tower['spec'],
                    'target_id': tower['target'],
                    'color': self.TOWER_COLORS[tower['type']]
                })
                # sfx: laser_shoot

    def _update_projectiles(self):
        reward = 0
        projectiles_to_remove = []
        for i, p in enumerate(self.projectiles):
            target_enemy = next((e for e in self.enemies if e['id'] == p['target_id']), None)

            if target_enemy is None:
                projectiles_to_remove.append(i)
                continue

            # Move towards target
            direction = target_enemy['pos'] - p['pos']
            dist = np.linalg.norm(direction)
            if dist < p['spec']['proj_speed']:
                # Hit
                projectiles_to_remove.append(i)
                reward += self._handle_projectile_hit(p, target_enemy)
            else:
                p['pos'] += (direction / dist) * p['spec']['proj_speed']
        
        # Remove projectiles that hit or lost their target
        for i in sorted(projectiles_to_remove, reverse=True):
            del self.projectiles[i]

        return reward

    def _handle_projectile_hit(self, projectile, main_target):
        hit_reward = 0
        spec = projectile['spec']
        impact_pos = main_target['pos']

        # Create impact particles
        self._create_particles(impact_pos, projectile['color'], 10, 2)
        # sfx: projectile_hit

        if spec['aoe'] > 0:
            # AoE damage
            for enemy in self.enemies:
                if np.linalg.norm(enemy['pos'] - impact_pos) <= spec['aoe']:
                    enemy['health'] -= spec['damage']
                    hit_reward += 0.1
        else:
            # Single target damage
            main_target['health'] -= spec['damage']
            hit_reward += 0.1
        
        return hit_reward

    def _update_enemies(self):
        reward = 0
        base_damage = 0
        enemies_to_remove = []
        
        for i, enemy in enumerate(self.enemies):
            # Health check
            if enemy['health'] <= 0:
                self.score += 10
                self.money += enemy['money_reward']
                reward += 1
                enemies_to_remove.append(i)
                self._create_particles(enemy['pos'], COLOR_HEALTH_BAR_ENEMY, 20, 3)
                # sfx: enemy_die
                continue

            # Movement
            if enemy['path_target_idx'] >= len(self.path_waypoints_px):
                self.base_health -= 10
                base_damage += 10
                reward -= 10
                enemies_to_remove.append(i)
                # sfx: base_damage
                continue

            target_pos = np.array(self.path_waypoints_px[enemy['path_target_idx']])
            direction = target_pos - enemy['pos']
            dist = np.linalg.norm(direction)

            if dist < enemy['speed']:
                enemy['path_target_idx'] += 1
            else:
                enemy['pos'] += (direction / dist) * enemy['speed']

        # Remove dead/finished enemies
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]

        self.base_health = max(0, self.base_health)
        return {'reward': reward, 'base_damage': base_damage}

    def _update_particles(self):
        particles_to_remove = []
        for i, p in enumerate(self.particles):
            p['pos'] += p['vel']
            p['vel'] *= 0.95 # Damping
            p['life'] -= 1
            if p['life'] <= 0:
                particles_to_remove.append(i)
        
        for i in sorted(particles_to_remove, reverse=True):
            del self.particles[i]

    def _update_effects(self):
        if self.screen_flash_alpha > 0:
            self.screen_flash_alpha = max(0, self.screen_flash_alpha - 15)

    def _create_particles(self, pos, color, count, max_speed):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, max_speed)
            self.particles.append({
                'pos': np.array(pos, dtype=float),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': self.np_random.integers(10, 25),
                'color': color,
                'radius': self.np_random.uniform(1, 3)
            })

    def _check_termination(self):
        win = self.wave_index >= self.TOTAL_WAVES and not self.enemies
        lose_health = self.base_health <= 0
        lose_time = self.steps >= self.MAX_STEPS
        return win or lose_health or lose_time

    # --- Rendering ---
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_path()
        self._render_placement_spots()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        self._render_effects()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "money": self.money,
            "wave": self.wave_index + 1,
            "base_health": self.base_health
        }

    def _render_path(self):
        if len(self.path_waypoints_px) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints_px, self.CELL_SIZE)

    def _render_placement_spots(self):
        cursor_pos = self.placement_spots[self.cursor_index]
        for pos in self.placement_spots:
            rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            color = self.COLOR_PLACE_TILE_HOVER if pos == cursor_pos else self.COLOR_PLACE_TILE
            pygame.draw.rect(self.screen, color, rect, 1, border_radius=2)

    def _render_base(self):
        base_pos = self.path_waypoints_grid[-1]
        rect = pygame.Rect(base_pos[0] * self.CELL_SIZE, base_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, rect, border_radius=3)

    def _render_towers(self):
        for tower in self.towers:
            x, y = tower['px_pos']
            color = self.TOWER_COLORS[tower['type']]
            points = [(x, y - 8), (x - 7, y + 5), (x + 7, y + 5)]
            pygame.gfxdraw.aapolygon(self.screen, [(int(px), int(py)) for px, py in points], color)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(px), int(py)) for px, py in points], color)

    def _render_enemies(self):
        for enemy in self.enemies:
            x, y = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.aacircle(self.screen, x, y, 7, self.COLOR_HEALTH_BAR_ENEMY)
            pygame.gfxdraw.filled_circle(self.screen, x, y, 7, self.COLOR_HEALTH_BAR_ENEMY)
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 16
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (x - bar_w/2, y - 14, bar_w, 4))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_PLAYER, (x - bar_w/2, y - 14, bar_w * health_pct, 4))

    def _render_projectiles(self):
        for p in self.projectiles:
            x, y = int(p['pos'][0]), int(p['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, x, y, 3, p['color'])

    def _render_particles(self):
        for p in self.particles:
            alpha = int(255 * (p['life'] / 25))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['radius']*2, p['radius']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['radius'], p['radius']), p['radius'])
            self.screen.blit(temp_surf, (int(p['pos'][0] - p['radius']), int(p['pos'][1] - p['radius'])))

    def _render_cursor(self):
        pos = self.placement_spots[self.cursor_index]
        rect = pygame.Rect(pos[0] * self.CELL_SIZE, pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_SCORE, rect, 2, border_radius=3)

    def _render_ui(self):
        # Wave Info
        if self.wave_timer > 0 and not self.enemies and self.wave_index < self.TOTAL_WAVES:
            wave_text = f"Wave {self.wave_index + 2} in {math.ceil(self.wave_timer / 30)}"
        elif self.wave_index >= self.TOTAL_WAVES and not self.enemies:
            wave_text = "YOU WIN!"
        elif self.base_health <= 0:
            wave_text = "GAME OVER"
        else:
            wave_text = f"Wave {self.wave_index + 1}/{self.TOTAL_WAVES}"
        
        surf = self.font_wave.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(surf, (15, 10))

        # Base Health
        health_surf = self.font_ui.render(f"Base: {self.base_health}/{self.BASE_START_HEALTH}", True, self.COLOR_TEXT)
        self.screen.blit(health_surf, (self.SCREEN_WIDTH - health_surf.get_width() - 15, 10))
        health_pct = self.base_health / self.BASE_START_HEALTH
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (self.SCREEN_WIDTH - 115, 35, 100, 8))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_PLAYER, (self.SCREEN_WIDTH - 115, 35, 100 * health_pct, 8))

        # Bottom UI Panel
        panel_y = self.SCREEN_HEIGHT - 40
        pygame.draw.rect(self.screen, self.COLOR_PATH, (0, panel_y, self.SCREEN_WIDTH, 40))
        
        # Money
        money_surf = self.font_ui.render(f"${self.money}", True, self.COLOR_SCORE)
        self.screen.blit(money_surf, (15, panel_y + 10))

        # Score
        score_surf = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (self.SCREEN_WIDTH - score_surf.get_width() - 15, panel_y + 10))

        # Selected Tower
        spec = self.TOWER_SPECS[self.selected_tower_type]
        cost_color = self.COLOR_SCORE if self.money >= spec['cost'] else self.COLOR_HEALTH_BAR_ENEMY
        tower_surf = self.font_ui.render(f"Build: {self.selected_tower_type.capitalize()} (${spec['cost']})", True, cost_color)
        self.screen.blit(tower_surf, (self.SCREEN_WIDTH / 2 - tower_surf.get_width() / 2, panel_y + 10))

    def _render_effects(self):
        if self.screen_flash_alpha > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill((239, 83, 80, int(self.screen_flash_alpha)))
            self.screen.blit(flash_surface, (0, 0))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")