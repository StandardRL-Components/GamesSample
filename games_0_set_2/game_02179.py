
# Generated: 2025-08-28T03:58:13.524377
# Source Brief: brief_02179.md
# Brief Index: 2179

        
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
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle tower types. Press Space to place a tower."
    )

    game_description = (
        "Defend your base from enemy waves by strategically placing towers on the grid. Survive all 5 waves to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 14
        self.MAX_STEPS = 30 * 120 # 2 minutes at 30fps
        self.MAX_WAVES = 5
        self.FPS = 30

        # --- Colors ---
        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PATH = (60, 65, 70)
        self.COLOR_START = (80, 150, 80)
        self.COLOR_END = (150, 80, 80)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_RANGE = (0, 100, 200, 100) # RGBA
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_UI_BG = (10, 10, 10, 180) # RGBA
        self.COLOR_ENEMY = (210, 40, 40)
        self.COLOR_HEALTH_BAR_BG = (80, 80, 80)
        self.COLOR_HEALTH_BAR = (40, 210, 40)

        # --- Isometric Projection ---
        self.TILE_WIDTH_ISO = 32
        self.TILE_HEIGHT_ISO = 16
        self.ORIGIN_X = self.SCREEN_WIDTH // 2
        self.ORIGIN_Y = 80

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
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.path = []
        self.path_tiles = set()
        self._generate_path()
        
        self.tower_types = [
            {'name': 'Cannon', 'cost': 25, 'range': 2.5, 'damage': 10, 'fire_rate': 20, 'color': (100, 100, 255)},
            {'name': 'Missile', 'cost': 60, 'range': 4.5, 'damage': 30, 'fire_rate': 60, 'color': (255, 150, 50)}
        ]
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.resources = 0
        self.wave_number = 0
        self.wave_enemies_to_spawn = []
        self.wave_spawn_timer = 0
        
        self.cursor_pos = [0, 0]
        self.selected_tower_index = 0
        self.last_action = np.array([0, 0, 0])
        self.cursor_move_cooldown = 0

        self.reset()
        self.validate_implementation()

    def _grid_to_screen(self, grid_x, grid_y):
        screen_x = self.ORIGIN_X + (grid_x - grid_y) * (self.TILE_WIDTH_ISO / 2)
        screen_y = self.ORIGIN_Y + (grid_x + grid_y) * (self.TILE_HEIGHT_ISO / 2)
        return int(screen_x), int(screen_y)

    def _generate_path(self):
        path_nodes = [(0, 6), (10, 6), (10, 2), (2, 2), (2, 11), (19, 11)]
        self.path = []
        for i in range(len(path_nodes) - 1):
            x1, y1 = path_nodes[i]
            x2, y2 = path_nodes[i+1]
            dx, dy = x2 - x1, y2 - y1
            steps = max(abs(dx), abs(dy))
            for j in range(steps):
                x = x1 + int(round(j * dx / steps))
                y = y1 + int(round(j * dy / steps))
                if not self.path or (x, y) != self.path[-1]:
                    self.path.append((x, y))
        self.path.append(path_nodes[-1])
        self.path_tiles = set(self.path)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition_met = False
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.resources = 80
        self.wave_number = 0
        self.wave_enemies_to_spawn = []
        self.wave_spawn_timer = 0
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_tower_index = 0
        self.last_action = np.array([0, 0, 0])
        self.cursor_move_cooldown = 0
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return

        num_enemies = 4 + self.wave_number * 2
        spawn_delay = max(20, 60 - self.wave_number * 5)
        base_health = 20 * (1.05 ** (self.wave_number - 1))
        base_speed = 0.02 * (1.05 ** (self.wave_number - 1))

        self.wave_enemies_to_spawn = []
        for i in range(num_enemies):
            enemy = {
                'path_index': 0,
                'sub_pos': 0.0,
                'max_health': base_health,
                'health': base_health,
                'speed': base_speed * self.np_random.uniform(0.9, 1.1),
                'id': self.np_random.integers(1, 1_000_000)
            }
            self.wave_enemies_to_spawn.append(enemy)
        
        self.wave_spawn_timer = spawn_delay

    def step(self, action):
        reward = 0
        terminated = False
        
        if not self.game_over:
            reward += self._handle_input(action)
            reward += self._update_game_state()
        
        self.steps += 1
        
        if self.game_over or self.steps >= self.MAX_STEPS:
            terminated = True
            if self.win_condition_met:
                # Add a final bonus for winning the whole game
                reward += 200
        
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _handle_input(self, action):
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1
        reward = 0
        tower_placed_this_step = False

        # --- Cursor Movement ---
        if self.cursor_move_cooldown > 0:
            self.cursor_move_cooldown -= 1
        
        if movement != 0 and self.cursor_move_cooldown == 0:
            if movement == 1: self.cursor_pos[1] -= 1
            elif movement == 2: self.cursor_pos[1] += 1
            elif movement == 3: self.cursor_pos[0] -= 1
            elif movement == 4: self.cursor_pos[0] += 1
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)
            self.cursor_move_cooldown = 3 # 3-frame cooldown

        # --- Cycle Tower (on press) ---
        if shift_press and not self.last_action[2]:
            self.selected_tower_index = (self.selected_tower_index + 1) % len(self.tower_types)
            # sfx: UI_cycle

        # --- Place Tower (on press) ---
        if space_press and not self.last_action[1]:
            tower_type = self.tower_types[self.selected_tower_index]
            can_afford = self.resources >= tower_type['cost']
            is_valid_tile = tuple(self.cursor_pos) not in self.path_tiles and not any(t['grid_pos'] == self.cursor_pos for t in self.towers)

            if can_afford and is_valid_tile:
                self.resources -= tower_type['cost']
                new_tower = {
                    'type_index': self.selected_tower_index,
                    'grid_pos': list(self.cursor_pos),
                    'cooldown': 0,
                    'target_id': None
                }
                self.towers.append(new_tower)
                tower_placed_this_step = True
                # sfx: place_tower
                for _ in range(20):
                    self._create_particle(self._grid_to_screen(*self.cursor_pos), self.COLOR_CURSOR, 1, 5, 20)

        if not tower_placed_this_step:
            reward -= 0.01

        self.last_action = action
        return reward

    def _update_game_state(self):
        reward = 0

        # --- Enemy Spawning ---
        if self.wave_spawn_timer > 0:
            self.wave_spawn_timer -= 1
        elif len(self.wave_enemies_to_spawn) > 0:
            self.enemies.append(self.wave_enemies_to_spawn.pop(0))
            spawn_delay = max(20, 60 - self.wave_number * 5)
            self.wave_spawn_timer = spawn_delay

        # --- Tower Logic ---
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
            
            tower_type = self.tower_types[tower['type_index']]
            if tower['cooldown'] == 0:
                # Find target
                possible_targets = []
                for enemy in self.enemies:
                    dist = math.dist(tower['grid_pos'], self.path[enemy['path_index']])
                    if dist <= tower_type['range']:
                        possible_targets.append(enemy)
                
                # Target enemy closest to the end of the path
                if possible_targets:
                    best_target = max(possible_targets, key=lambda e: e['path_index'] + e['sub_pos'])
                    tower['target_id'] = best_target['id']
                    
                    # Fire projectile
                    start_pos = self._grid_to_screen(*tower['grid_pos'])
                    self.projectiles.append({
                        'pos': list(start_pos),
                        'target_id': best_target['id'],
                        'damage': tower_type['damage'],
                        'speed': 8,
                        'type_index': tower['type_index']
                    })
                    tower['cooldown'] = tower_type['fire_rate']
                    # sfx: tower_fire
                    self._create_particle(start_pos, tower_type['color'], 3, 8, 8, count=5)

        # --- Projectile Logic ---
        for proj in self.projectiles[:]:
            target_enemy = next((e for e in self.enemies if e['id'] == proj['target_id']), None)
            if not target_enemy:
                self.projectiles.remove(proj)
                continue
            
            target_pos = self._grid_to_screen(*self.path[target_enemy['path_index']])
            target_pos = (target_pos[0], target_pos[1] - self.TILE_HEIGHT_ISO) # Aim for body
            
            angle = math.atan2(target_pos[1] - proj['pos'][1], target_pos[0] - proj['pos'][0])
            proj['pos'][0] += math.cos(angle) * proj['speed']
            proj['pos'][1] += math.sin(angle) * proj['speed']

            if math.dist(proj['pos'], target_pos) < proj['speed']:
                target_enemy['health'] -= proj['damage']
                reward += 0.1
                self.projectiles.remove(proj)
                # sfx: enemy_hit
                self._create_particle(target_pos, self.tower_types[proj['type_index']]['color'], 2, 6, 15, count=15)

        # --- Enemy Logic ---
        for enemy in self.enemies[:]:
            if enemy['health'] <= 0:
                reward += 1.0
                self.score += 10
                self.resources += 5
                enemy_pos = self._grid_to_screen(*self.path[enemy['path_index']])
                self._create_particle(enemy_pos, self.COLOR_ENEMY, 3, 10, 25, count=30)
                self.enemies.remove(enemy)
                # sfx: enemy_death
                continue

            enemy['sub_pos'] += enemy['speed']
            if enemy['sub_pos'] >= 1.0:
                enemy['sub_pos'] -= 1.0
                enemy['path_index'] += 1
            
            if enemy['path_index'] >= len(self.path) - 1:
                self.game_over = True
                self.win_condition_met = False
                reward -= 100
                # sfx: game_over_lose
                self.enemies.remove(enemy)
                continue

        # --- Particle Logic ---
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['size'] = max(0, p['size'] * 0.95)
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- Wave Completion ---
        if not self.enemies and not self.wave_enemies_to_spawn and not self.game_over:
            if self.wave_number >= self.MAX_WAVES:
                self.game_over = True
                self.win_condition_met = True
                # sfx: game_over_win
            else:
                reward += 100
                self.score += 50 * self.wave_number
                self._start_next_wave()
                # sfx: wave_complete
        
        return reward

    def _create_particle(self, pos, color, min_size, max_size, life, count=1):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'color': color,
                'size': self.np_random.uniform(min_size, max_size),
                'life': life
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw Grid ---
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                screen_pos = self._grid_to_screen(x, y)
                is_path = (x, y) in self.path_tiles
                color = self.COLOR_PATH if is_path else self.COLOR_GRID
                
                points = [
                    self._grid_to_screen(x, y),
                    self._grid_to_screen(x + 1, y),
                    self._grid_to_screen(x + 1, y + 1),
                    self._grid_to_screen(x, y + 1)
                ]
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # --- Draw Start/End ---
        start_points = [self._grid_to_screen(0,6), self._grid_to_screen(1,6), self._grid_to_screen(1,7), self._grid_to_screen(0,7)]
        end_points = [self._grid_to_screen(19,11), self._grid_to_screen(20,11), self._grid_to_screen(20,12), self._grid_to_screen(19,12)]
        pygame.gfxdraw.filled_polygon(self.screen, start_points, self.COLOR_START)
        pygame.gfxdraw.filled_polygon(self.screen, end_points, self.COLOR_END)

        # --- Draw Cursor and Range ---
        cursor_screen_pos = self._grid_to_screen(*self.cursor_pos)
        cursor_points = [
            self._grid_to_screen(self.cursor_pos[0], self.cursor_pos[1]),
            self._grid_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1]),
            self._grid_to_screen(self.cursor_pos[0] + 1, self.cursor_pos[1] + 1),
            self._grid_to_screen(self.cursor_pos[0], self.cursor_pos[1] + 1)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_CURSOR, cursor_points, 2)
        
        tower_type = self.tower_types[self.selected_tower_index]
        range_px = tower_type['range'] * (self.TILE_WIDTH_ISO / 2 + self.TILE_HEIGHT_ISO / 2) / 2
        pygame.gfxdraw.filled_circle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], int(range_px), self.COLOR_RANGE)
        pygame.gfxdraw.aacircle(self.screen, cursor_screen_pos[0], cursor_screen_pos[1], int(range_px), self.COLOR_RANGE)

        # --- Draw Towers ---
        for tower in self.towers:
            pos = self._grid_to_screen(*tower['grid_pos'])
            tower_type = self.tower_types[tower['type_index']]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 8, (60,60,60))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], 6, tower_type['color'])

        # --- Draw Enemies ---
        for enemy in sorted(self.enemies, key=lambda e: e['path_index'] + e['sub_pos']):
            if enemy['path_index'] >= len(self.path) - 1: continue
            
            p1 = self._grid_to_screen(*self.path[enemy['path_index']])
            p2 = self._grid_to_screen(*self.path[enemy['path_index'] + 1])
            
            x = p1[0] + (p2[0] - p1[0]) * enemy['sub_pos']
            y = p1[1] + (p2[1] - p1[1]) * enemy['sub_pos'] - self.TILE_HEIGHT_ISO / 2
            
            # Body
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 6, (255, 100, 100))
            
            # Health bar
            health_pct = enemy['health'] / enemy['max_health']
            bar_w = 12
            bar_x = int(x - bar_w / 2)
            bar_y = int(y - 12)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_w, 3))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, int(bar_w * health_pct), 3))

        # --- Draw Projectiles ---
        for proj in self.projectiles:
            color = self.tower_types[proj['type_index']]['color']
            pygame.draw.circle(self.screen, color, proj['pos'], 3)

        # --- Draw Particles ---
        for p in self.particles:
            pygame.draw.circle(self.screen, p['color'], p['pos'], max(1, int(p['size'])))

    def _render_ui(self):
        # --- UI Panel ---
        ui_panel = pygame.Surface((self.SCREEN_WIDTH, 60), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 0))

        # --- Texts ---
        wave_text = self.font_medium.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        res_text = self.font_medium.render(f"RESOURCES: {self.resources}", True, self.COLOR_TEXT)
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        
        self.screen.blit(wave_text, (10, 5))
        self.screen.blit(res_text, (10, 30))
        self.screen.blit(score_text, (200, 5))

        # --- Selected Tower Info ---
        tower_type = self.tower_types[self.selected_tower_index]
        name_text = self.font_medium.render(f"Selected: {tower_type['name']}", True, tower_type['color'])
        cost_text = self.font_small.render(f"Cost: {tower_type['cost']}", True, self.COLOR_TEXT)
        self.screen.blit(name_text, (self.SCREEN_WIDTH - 200, 5))
        self.screen.blit(cost_text, (self.SCREEN_WIDTH - 200, 35))

        # --- Game Over/Win Message ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_condition_met else "GAME OVER"
            color = (100, 255, 100) if self.win_condition_met else (255, 100, 100)
            
            text = self.font_large.render(message, True, color)
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "towers": len(self.towers),
            "enemies": len(self.enemies),
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")