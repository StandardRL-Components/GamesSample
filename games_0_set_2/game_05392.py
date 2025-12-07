
# Generated: 2025-08-28T04:52:30.952615
# Source Brief: brief_05392.md
# Brief Index: 5392

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the cursor. Press Space to build/upgrade towers or start the wave."
    )

    game_description = (
        "Defend your base from waves of enemies in this isometric tower defense game. "
        "Place towers on the grid to protect your base. Earn gold by defeating enemies and surviving waves."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (30, 35, 40)
    COLOR_GRID = (50, 55, 60)
    COLOR_PATH = (60, 65, 70)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_DMG = (200, 50, 50)
    COLOR_TOWER_L1 = (40, 200, 100)
    COLOR_TOWER_L2 = (255, 200, 50)
    COLOR_TOWER_L3 = (255, 100, 150)
    COLOR_ENEMY_A = (220, 50, 50)
    COLOR_PROJ = (255, 255, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_GOLD = (255, 223, 0)
    
    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 14, 10
    TILE_WIDTH_HALF = 24
    TILE_HEIGHT_HALF = 12
    ISO_Z_SCALE = 18 # Height of isometric cubes
    
    # Game Parameters
    MAX_STEPS = 3000
    MAX_WAVES = 10
    INITIAL_GOLD = 80
    INITIAL_BASE_HEALTH = 100
    WAVE_GOLD_BONUS = 25
    
    # Tower Stats
    TOWER_COSTS = [50, 100, 200]
    TOWER_STATS = {
        1: {"range": 80, "damage": 10, "fire_rate": 1.0, "color": COLOR_TOWER_L1},
        2: {"range": 90, "damage": 25, "fire_rate": 0.8, "color": COLOR_TOWER_L2},
        3: {"range": 100, "damage": 50, "fire_rate": 0.6, "color": COLOR_TOWER_L3},
    }

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_large = pygame.font.Font(None, 48)
        
        self.grid_origin = (self.SCREEN_WIDTH // 2, 80)
        self.path_grid_coords = self._define_path()
        self.path_world_coords = [self._iso_to_world(x, y, 0) for x, y in self.path_grid_coords]
        self.buildable_tiles = self._get_buildable_tiles()

        self.reset()
        
        # This can be commented out for performance after verification
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.game_phase = "WAVE_PREP"
        self.wave_number = 1
        self.gold = self.INITIAL_GOLD
        self.base_health = self.INITIAL_BASE_HEALTH
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        self.wave_spawn_list = []
        self.wave_spawn_timer = 0
        self.wave_fully_spawned = False
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.action_cooldown = 0
        
        self.start_button_rect = pygame.Rect(500, 350, 120, 40)
        self.cursor_on_button = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        
        self._handle_actions(action)
        
        if self.game_phase == "WAVE_ACTIVE":
            reward += self._update_game_state()
        
        self.score += reward
        terminated = self._check_termination()

        if terminated and not self.win:
            reward -= 100 # Loss penalty
        elif terminated and self.win:
            reward += 100 # Win bonus
        
        self.score += reward # Add terminal reward to score
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_actions(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        if self.action_cooldown > 0:
            self.action_cooldown -= 1
        
        if self.game_phase == "WAVE_PREP" and self.action_cooldown == 0:
            # --- Cursor Movement ---
            dx, dy = 0, 0
            if movement == 1: dy = -1
            elif movement == 2: dy = 1
            elif movement == 3: dx = -1
            elif movement == 4: dx = 1

            if dx != 0 or dy != 0:
                self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_WIDTH - 1)
                self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_HEIGHT - 1)
                self.action_cooldown = 3 # Debounce movement
            
            # --- Interaction ---
            if space_held:
                if self.cursor_on_button: # Start Wave
                    self.game_phase = "WAVE_ACTIVE"
                    self._prepare_next_wave()
                    # sound: wave_start.wav
                else: # Build/Upgrade Tower
                    self._try_build_or_upgrade()
                self.action_cooldown = 10 # Debounce spacebar

    def _update_game_state(self):
        reward = 0
        
        # Spawn enemies
        self.wave_spawn_timer -= 1
        if self.wave_spawn_timer <= 0 and self.wave_spawn_list:
            self.enemies.append(self.wave_spawn_list.pop(0))
            self.wave_spawn_timer = 30 # Spawn every 1 second at 30fps
        if not self.wave_spawn_list:
            self.wave_fully_spawned = True
            
        # Update towers
        for tower in self.towers:
            tower['cooldown'] = max(0, tower['cooldown'] - 1)
            if tower['cooldown'] == 0:
                target = self._find_target(tower)
                if target:
                    self._fire_projectile(tower, target)
                    tower['cooldown'] = tower['fire_rate'] * 30 # in frames
                    # sound: tower_fire.wav
        
        # Update projectiles
        for p in self.projectiles[:]:
            p['pos'] = (p['pos'][0] + p['vel'][0], p['pos'][1] + p['vel'][1])
            dist_sq = (p['pos'][0] - p['target']['pos'][0])**2 + (p['pos'][1] - p['target']['pos'][1])**2
            if dist_sq < 100: # Hit
                p['target']['health'] -= p['damage']
                self._create_particles(p['pos'], self.COLOR_PROJ, 5)
                if p['target']['health'] <= 0:
                    reward += 0.1 # Kill reward
                    self.gold += p['target']['value']
                    self._create_particles(p['target']['pos'], self.COLOR_ENEMY_A, 15)
                    self.enemies.remove(p['target'])
                    # sound: enemy_die.wav
                self.projectiles.remove(p)
                # sound: enemy_hit.wav
        
        # Update enemies
        for enemy in self.enemies[:]:
            path_idx = enemy['path_index']
            if path_idx >= len(self.path_world_coords):
                continue
            
            target_pos = self.path_world_coords[path_idx]
            dx, dy = target_pos[0] - enemy['pos'][0], target_pos[1] - enemy['pos'][1]
            dist = math.hypot(dx, dy)
            
            if dist < enemy['speed']:
                enemy['path_index'] += 1
                if enemy['path_index'] >= len(self.path_world_coords):
                    self.base_health -= 10
                    reward -= 0.1 # 10 damage * 0.01
                    self.enemies.remove(enemy)
                    # sound: base_damage.wav
                    continue
            else:
                enemy['pos'] = (enemy['pos'][0] + dx/dist * enemy['speed'], enemy['pos'][1] + dy/dist * enemy['speed'])
        
        # Update particles
        for particle in self.particles[:]:
            particle['pos'] = (particle['pos'][0] + particle['vel'][0], particle['pos'][1] + particle['vel'][1])
            particle['lifespan'] -= 1
            if particle['lifespan'] <= 0:
                self.particles.remove(particle)
        
        # Check for wave end
        if self.wave_fully_spawned and not self.enemies:
            self.game_phase = "WAVE_PREP"
            self.wave_number += 1
            self.gold += self.WAVE_GOLD_BONUS + (self.wave_number - 1) * 5
            reward += 1.0 # Wave complete reward
            # sound: wave_complete.wav
            if self.wave_number > self.MAX_WAVES:
                self.win = True
        
        return reward
    
    def _check_termination(self):
        if self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.win:
            self.game_over = True
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "gold": self.gold, "wave": self.wave_number}

    # --- Rendering ---
    def _render_game(self):
        # Path
        if len(self.path_world_coords) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_world_coords, 10)
        
        # Grid
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                p1 = self._iso_to_world(x, y, 0)
                p2 = self._iso_to_world(x + 1, y, 0)
                p3 = self._iso_to_world(x + 1, y + 1, 0)
                p4 = self._iso_to_world(x, y + 1, 0)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p4)

        # Base (end of path)
        base_pos = self.path_world_coords[-1]
        base_color = self.COLOR_BASE if self.base_health > 30 else self.COLOR_BASE_DMG
        self._draw_iso_cube(self.screen, base_pos, self.TILE_WIDTH_HALF * 2, self.ISO_Z_SCALE * 1.5, base_color)
        
        # Towers
        for tower in self.towers:
            pos = self._iso_to_world(tower['pos'][0], tower['pos'][1], 0)
            self._draw_iso_cube(self.screen, pos, self.TILE_WIDTH_HALF * 1.5, self.ISO_Z_SCALE, tower['color'])
            if tower['level'] > 1:
                self._draw_iso_cube(self.screen, (pos[0], pos[1] - self.ISO_Z_SCALE), self.TILE_WIDTH_HALF * 1.2, self.ISO_Z_SCALE * 0.8, tower['color'])
            if tower['level'] > 2:
                self._draw_iso_cube(self.screen, (pos[0], pos[1] - self.ISO_Z_SCALE * 1.8), self.TILE_WIDTH_HALF * 0.9, self.ISO_Z_SCALE * 0.6, tower['color'])

        # Cursor
        if self.game_phase == "WAVE_PREP":
            self._render_cursor()

        # Render entities (sorted by y-pos for correct layering)
        render_list = self.enemies + self.towers # Simplified; a full implementation would sort all objects
        
        # Enemies
        for enemy in self.enemies:
            self._draw_iso_cube(self.screen, enemy['pos'], self.TILE_WIDTH_HALF, self.ISO_Z_SCALE*0.8, self.COLOR_ENEMY_A, True)
            # Health bar
            hb_width = 20
            hb_height = 4
            hb_pos = (enemy['pos'][0] - hb_width/2, enemy['pos'][1] - self.ISO_Z_SCALE - 10)
            health_ratio = enemy['health'] / enemy['max_health']
            pygame.draw.rect(self.screen, (100,0,0), (*hb_pos, hb_width, hb_height))
            pygame.draw.rect(self.screen, (0,200,0), (*hb_pos, hb_width * health_ratio, hb_height))

        # Projectiles and Particles
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, p['color'])
        
        for part in self.particles:
            alpha = int(255 * (part['lifespan'] / part['max_lifespan']))
            color = (*part['color'], alpha)
            pygame.gfxdraw.filled_circle(self.screen, int(part['pos'][0]), int(part['pos'][1]), int(part['radius']), color)

    def _render_ui(self):
        # Top bar
        pygame.draw.rect(self.screen, (0,0,0,150), (0, 0, self.SCREEN_WIDTH, 40))
        
        # Gold
        gold_text = self.font_medium.render(f"GOLD: {self.gold}", True, self.COLOR_GOLD)
        self.screen.blit(gold_text, (10, 10))
        
        # Wave
        wave_text = self.font_medium.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH/2 - wave_text.get_width()/2, 10))
        
        # Base Health
        health_text = self.font_medium.render(f"BASE HP: {self.base_health}", True, self.COLOR_BASE)
        self.screen.blit(health_text, (self.SCREEN_WIDTH - health_text.get_width() - 10, 10))
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 100, 100)
            end_text = self.font_large.render(msg, True, color)
            self.screen.blit(end_text, (self.SCREEN_WIDTH/2 - end_text.get_width()/2, self.SCREEN_HEIGHT/2 - end_text.get_height()/2))

        # Prep phase UI
        if self.game_phase == "WAVE_PREP":
            # Start button
            pygame.draw.rect(self.screen, (0, 150, 0), self.start_button_rect, border_radius=5)
            start_text = self.font_medium.render("START WAVE", True, self.COLOR_TEXT)
            self.screen.blit(start_text, (self.start_button_rect.centerx - start_text.get_width()/2, self.start_button_rect.centery - start_text.get_height()/2))
            
            # Cursor info
            self._render_cursor_info()

    def _render_cursor(self):
        cursor_world_pos = self._iso_to_world(self.cursor_pos[0], self.cursor_pos[1], 0)
        
        # Check if cursor is on the start button
        self.cursor_on_button = self.start_button_rect.collidepoint(cursor_world_pos)
        
        if self.cursor_on_button:
            pygame.draw.rect(self.screen, (255, 255, 0), self.start_button_rect, 3, border_radius=5)
        else:
            # Draw cursor on grid
            x, y = self.cursor_pos
            p1 = self._iso_to_world(x, y, 0)
            p2 = self._iso_to_world(x + 1, y, 0)
            p3 = self._iso_to_world(x + 1, y + 1, 0)
            p4 = self._iso_to_world(x, y + 1, 0)
            
            status = self._get_cursor_status()
            color = (255, 255, 255) # Default
            if status == "build": color = (0, 255, 0)
            elif status == "upgrade": color = (0, 150, 255)
            elif status == "invalid": color = (255, 0, 0)
            
            pygame.draw.polygon(self.screen, color, [p1, p2, p3, p4], 2)

    def _render_cursor_info(self):
        status = self._get_cursor_status()
        info_text = ""
        cost = 0
        if status == "build":
            cost = self.TOWER_COSTS[0]
            info_text = f"Build Tower (Cost: {cost})"
        elif status == "upgrade":
            tower = next((t for t in self.towers if t['pos'] == self.cursor_pos), None)
            if tower and tower['level'] < 3:
                cost = self.TOWER_COSTS[tower['level']]
                info_text = f"Upgrade Tower (Cost: {cost})"
            else:
                info_text = "Max Level"
        elif status == "invalid":
            info_text = "Cannot Build Here"

        if info_text:
            color = self.COLOR_TEXT if not cost or self.gold >= cost else (255, 100, 100)
            text_surf = self.font_small.render(info_text, True, color)
            pos = (10, self.SCREEN_HEIGHT - text_surf.get_height() - 10)
            self.screen.blit(text_surf, pos)
            
    # --- Helper Functions ---
    def _iso_to_world(self, x, y, z):
        iso_x = self.grid_origin[0] + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.grid_origin[1] + (x + y) * self.TILE_HEIGHT_HALF - z
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, surface, pos, size, height, color, is_enemy=False):
        x, y = pos
        w_half, h_half = size / 2, size / 4
        
        top_points = [
            (x, y - height),
            (x + w_half, y - h_half - height),
            (x, y - 2 * h_half - height),
            (x - w_half, y - h_half - height),
        ]
        left_points = [
            (x - w_half, y - h_half - height),
            (x, y - 2 * h_half - height),
            (x, y - 2 * h_half),
            (x - w_half, y - h_half),
        ]
        right_points = [
            (x + w_half, y - h_half - height),
            (x, y - 2 * h_half - height),
            (x, y - 2 * h_half),
            (x + w_half, y - h_half),
        ]

        darker = tuple(max(0, c - 40) for c in color)
        darkest = tuple(max(0, c - 60) for c in color)
        
        pygame.gfxdraw.filled_polygon(surface, top_points, color)
        if not is_enemy: pygame.gfxdraw.aapolygon(surface, top_points, color)
        
        pygame.gfxdraw.filled_polygon(surface, left_points, darker)
        if not is_enemy: pygame.gfxdraw.aapolygon(surface, left_points, darker)
        
        pygame.gfxdraw.filled_polygon(surface, right_points, darkest)
        if not is_enemy: pygame.gfxdraw.aapolygon(surface, right_points, darkest)

    def _define_path(self):
        return [(0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (7, 6), (8, 6), (9, 6), (10, 6), (10, 5), (10, 4), (10, 3), (11, 3), (12, 3), (13, 3)]

    def _get_buildable_tiles(self):
        buildable = set()
        path_set = set(map(tuple, self.path_grid_coords))
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in path_set:
                    buildable.add((x, y))
        return buildable

    def _prepare_next_wave(self):
        self.wave_spawn_list = []
        self.wave_spawn_timer = 60 # 2 sec delay before first spawn
        self.wave_fully_spawned = False
        
        num_enemies = 5 + self.wave_number * 2
        health = 20 * (1 + (self.wave_number - 1) * 0.15)
        speed = 1.0 * (1 + (self.wave_number - 1) * 0.05)
        value = 5 + self.wave_number
        
        start_pos = self.path_world_coords[0]
        
        for i in range(num_enemies):
            enemy = {
                'pos': (start_pos[0] + self.np_random.uniform(-5,5), start_pos[1] + self.np_random.uniform(-5,5)),
                'path_index': 0,
                'health': health,
                'max_health': health,
                'speed': speed,
                'value': value,
            }
            self.wave_spawn_list.append(enemy)

    def _get_cursor_status(self):
        pos_tuple = tuple(self.cursor_pos)
        if pos_tuple not in self.buildable_tiles:
            return "invalid"
        
        existing_tower = next((t for t in self.towers if tuple(t['pos']) == pos_tuple), None)
        if existing_tower:
            return "upgrade"
        else:
            return "build"

    def _try_build_or_upgrade(self):
        status = self._get_cursor_status()
        if status == "build":
            cost = self.TOWER_COSTS[0]
            if self.gold >= cost:
                self.gold -= cost
                stats = self.TOWER_STATS[1]
                new_tower = {
                    'pos': list(self.cursor_pos), 'level': 1,
                    'range': stats['range'], 'damage': stats['damage'],
                    'fire_rate': stats['fire_rate'], 'color': stats['color'],
                    'cooldown': 0,
                }
                self.towers.append(new_tower)
                # sound: build_tower.wav
        elif status == "upgrade":
            tower = next((t for t in self.towers if t['pos'] == self.cursor_pos), None)
            if tower and tower['level'] < 3:
                cost = self.TOWER_COSTS[tower['level']]
                if self.gold >= cost:
                    self.gold -= cost
                    tower['level'] += 1
                    stats = self.TOWER_STATS[tower['level']]
                    tower.update(stats)
                    # sound: upgrade_tower.wav

    def _find_target(self, tower):
        tower_pos = self._iso_to_world(tower['pos'][0], tower['pos'][1], 0)
        for enemy in self.enemies:
            dist_sq = (tower_pos[0] - enemy['pos'][0])**2 + (tower_pos[1] - enemy['pos'][1])**2
            if dist_sq < tower['range']**2:
                return enemy
        return None

    def _fire_projectile(self, tower, target):
        start_pos = self._iso_to_world(tower['pos'][0], tower['pos'][1], self.ISO_Z_SCALE * 1.5)
        dx = target['pos'][0] - start_pos[0]
        dy = target['pos'][1] - start_pos[1]
        dist = math.hypot(dx, dy)
        speed = 10
        vel = (dx/dist * speed, dy/dist * speed) if dist > 0 else (0,0)
        
        proj = {
            'pos': start_pos, 'vel': vel, 'damage': tower['damage'], 'target': target, 'color': self.COLOR_PROJ
        }
        self.projectiles.append(proj)
    
    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            lifespan = self.np_random.integers(10, 20)
            self.particles.append({
                'pos': list(pos),
                'vel': (math.cos(angle) * speed, math.sin(angle) * speed),
                'radius': self.np_random.uniform(2, 5),
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan
            })

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
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")