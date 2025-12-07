
# Generated: 2025-08-27T21:40:13.284218
# Source Brief: brief_02870.md
# Brief Index: 2870

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the placement cursor. "
        "Press Space to build a tower. Press Shift to cycle tower types."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric tower defense game. Place towers to defend your base "
        "from waves of enemies. Survive all three waves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Colors ---
    COLOR_BG = (25, 30, 35)
    COLOR_PATH = (50, 60, 70)
    COLOR_GRID = (70, 80, 90)
    COLOR_GRID_HOVER = (255, 255, 0)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_STROKE = (100, 220, 255)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_ENEMY_STROKE = (255, 120, 120)
    COLOR_PROJECTILE_CANNON = (255, 200, 0)
    COLOR_PROJECTILE_GATLING = (255, 255, 150)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_HEALTH_BG = (100, 0, 0)
    COLOR_HEALTH_FG = (0, 200, 0)

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 3000
    TOTAL_WAVES = 3

    BASE_HEALTH_MAX = 100
    ISO_OFFSET_X = SCREEN_WIDTH // 2
    ISO_OFFSET_Y = SCREEN_HEIGHT // 5
    TILE_WIDTH = 28
    TILE_HEIGHT = 14

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_huge = pygame.font.SysFont("monospace", 48, bold=True)

        self.tower_definitions = {
            0: {"name": "Cannon", "cost": 10, "range": 100, "damage": 12, "fire_rate": 1.0, "color": (180, 180, 180)},
            1: {"name": "Gatling", "cost": 15, "range": 70, "damage": 4, "fire_rate": 0.25, "color": (140, 140, 140)}
        }
        
        # This will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.money = 0
        self.current_wave = 0
        self.wave_countdown = 0
        self.spawn_timer = 0
        self.enemies_in_wave = 0
        self.enemies_to_spawn = deque()
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.previous_space_held = False
        self.previous_shift_held = False
        self.previous_move_action = 0
        self.move_cooldown = 0
        self.step_reward = 0.0

        self._init_map()
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def _init_map(self):
        # Define path in cartesian grid coordinates
        self.path_cart = [
            (-9, -2), (-8, -2), (-7, -2), (-6, -2), (-5, -2), (-4, -2), (-3, -2),
            (-3, -1), (-3, 0), (-3, 1), (-3, 2), (-3, 3), (-2, 3), (-1, 3), (0, 3),
            (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0), (3, -1), (3, -2),
            (4, -2), (5, -2), (6, -2), (7, -2), (8, -2), (9, -2), (10, -2)
        ]
        self.path_world = [self._cart_to_world(x, y) for x, y in self.path_cart]

        # Define tower placement grid
        self.grid_size = (5, 5)
        self.grid_offset = (-2, -2)
        self.placement_grid = []
        for gx in range(self.grid_size[0]):
            for gy in range(self.grid_size[1]):
                cart_pos = (gx + self.grid_offset[0], gy + self.grid_offset[1])
                if cart_pos not in self.path_cart and cart_pos != (0,0): # Cannot build on path or base
                    self.placement_grid.append(cart_pos)
        self.cursor_pos = [self.grid_size[0] // 2, self.grid_size[1] // 2]


    def _cart_to_iso(self, x, y):
        return (x - y), (x + y) / 2

    def _cart_to_world(self, x, y):
        iso_x, iso_y = self._cart_to_iso(x * self.TILE_WIDTH, y * self.TILE_WIDTH)
        return self.ISO_OFFSET_X + iso_x, self.ISO_OFFSET_Y + iso_y

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = self.BASE_HEALTH_MAX
        self.money = 40
        self.current_wave = 0
        self.wave_countdown = self.FPS * 3 # 3 seconds to start
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        self.enemies_to_spawn.clear()
        
        self.cursor_pos = [self.grid_size[0] // 2, self.grid_size[1] // 2]
        self.selected_tower_type = 0
        self.previous_space_held = True # Prevent action on first frame
        self.previous_shift_held = True
        self.previous_move_action = 0
        self.move_cooldown = 0
        
        return self._get_observation(), self._get_info()

    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.TOTAL_WAVES:
            return # Game won
            
        self.wave_countdown = 0
        self.enemies_in_wave = 5 + (self.current_wave - 1) * 3
        self.spawn_timer = self.FPS // 2

        base_health = 10 * (1.1 ** (self.current_wave - 1))
        base_speed = 1.0 * (1.1 ** (self.current_wave - 1))
        
        for _ in range(self.enemies_in_wave):
            self.enemies_to_spawn.append({'health': base_health, 'speed': base_speed})

    def step(self, action):
        self.step_reward = -0.001 # Small penalty for existing
        
        self._handle_input(action)

        if not self.game_over:
            self._update_game_state()

        self.steps += 1
        reward = self.step_reward
        terminated = self._check_termination()
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement with Cooldown ---
        if self.move_cooldown > 0:
            self.move_cooldown -= 1
        elif movement != 0:
            self.move_cooldown = 5 # 5 frames cooldown
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
            elif movement == 2: self.cursor_pos[1] = min(self.grid_size[1] - 1, self.cursor_pos[1] + 1) # Down
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
            elif movement == 4: self.cursor_pos[0] = min(self.grid_size[0] - 1, self.cursor_pos[0] + 1) # Right

        # --- Place Tower (rising edge) ---
        if space_held and not self.previous_space_held:
            self._place_tower()
        self.previous_space_held = space_held

        # --- Cycle Tower Type (rising edge) ---
        if shift_held and not self.previous_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.tower_definitions)
            # sfx: UI_cycle_sound
        self.previous_shift_held = shift_held

    def _place_tower(self):
        grid_x, grid_y = self.cursor_pos
        cart_pos = (grid_x + self.grid_offset[0], grid_y + self.grid_offset[1])
        
        if cart_pos not in self.placement_grid:
            # sfx: UI_error_sound
            return

        # Check if a tower is already there
        if any(t['cart_pos'] == cart_pos for t in self.towers):
            # sfx: UI_error_sound
            return
            
        tower_def = self.tower_definitions[self.selected_tower_type]
        if self.money >= tower_def['cost']:
            self.money -= tower_def['cost']
            world_pos = self._cart_to_world(*cart_pos)
            self.towers.append({
                'cart_pos': cart_pos,
                'world_pos': world_pos,
                'type': self.selected_tower_type,
                'cooldown': 0,
                'target': None,
            })
            # sfx: tower_place_sound
            self.step_reward += 0.1 # Small reward for building
        else:
            # sfx: UI_error_sound
            pass

    def _update_game_state(self):
        # --- Wave Management ---
        if self.wave_countdown > 0:
            self.wave_countdown -= 1
            if self.wave_countdown == 0:
                self._start_next_wave()
        
        elif len(self.enemies_to_spawn) > 0:
            self.spawn_timer -= 1
            if self.spawn_timer <= 0:
                self.spawn_timer = self.FPS // 2
                enemy_data = self.enemies_to_spawn.popleft()
                self.enemies.append({
                    'pos': list(self.path_world[0]),
                    'path_index': 0,
                    'health': enemy_data['health'],
                    'max_health': enemy_data['health'],
                    'speed': enemy_data['speed'],
                    'dist_to_next': 0,
                })
        
        elif len(self.enemies) == 0 and self.current_wave > 0 and self.current_wave <= self.TOTAL_WAVES:
            self.score += 50
            self.step_reward += 50
            self.money += 50 + self.current_wave * 10
            if self.current_wave == self.TOTAL_WAVES:
                self.game_over = True # Win
            else:
                self.wave_countdown = self.FPS * 5 # 5 seconds between waves
        
        # --- Update Entities ---
        self._update_towers()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()

    def _update_towers(self):
        for tower in self.towers:
            if tower['cooldown'] > 0:
                tower['cooldown'] -= 1
                continue
            
            tower_def = self.tower_definitions[tower['type']]
            target = None
            best_dist = float('inf')

            # Find closest enemy in range
            for enemy in self.enemies:
                dist = math.hypot(enemy['pos'][0] - tower['world_pos'][0], enemy['pos'][1] - tower['world_pos'][1])
                if dist < tower_def['range'] and dist < best_dist:
                    best_dist = dist
                    target = enemy
            
            if target:
                tower['cooldown'] = tower_def['fire_rate'] * self.FPS
                tower['target'] = target
                # sfx: tower_fire_sound
                self.projectiles.append({
                    'pos': list(tower['world_pos']),
                    'target': target,
                    'type': tower['type'],
                    'damage': tower_def['damage'],
                })

    def _update_enemies(self):
        for enemy in reversed(self.enemies):
            path_idx = enemy['path_index']
            if path_idx >= len(self.path_world) - 1:
                # Reached base
                self.base_health = max(0, self.base_health - enemy['health'])
                self.step_reward -= 10
                self.enemies.remove(enemy)
                # sfx: base_damage_sound
                self._create_explosion(self._cart_to_world(0,0), self.COLOR_BASE, 30)
                continue

            start_pos = self.path_world[path_idx]
            end_pos = self.path_world[path_idx + 1]
            
            vec = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            dist = math.hypot(*vec)
            if dist == 0:
                enemy['path_index'] += 1
                continue

            move_dist = enemy['speed']
            enemy['pos'][0] += (vec[0] / dist) * move_dist
            enemy['pos'][1] += (vec[1] / dist) * move_dist
            
            # Check if overshot waypoint
            if math.hypot(enemy['pos'][0] - start_pos[0], enemy['pos'][1] - start_pos[1]) >= dist:
                enemy['path_index'] += 1
                enemy['pos'] = list(end_pos)

    def _update_projectiles(self):
        for p in reversed(self.projectiles):
            if p['target'] not in self.enemies:
                self.projectiles.remove(p)
                continue
            
            target_pos = p['target']['pos']
            proj_pos = p['pos']
            
            vec = (target_pos[0] - proj_pos[0], target_pos[1] - proj_pos[1])
            dist = math.hypot(*vec)
            
            speed = 12 if p['type'] == 0 else 18 # Cannon slower, Gatling faster
            
            if dist < speed:
                # Hit target
                p['target']['health'] -= p['damage']
                self.step_reward += 0.1
                # sfx: enemy_hit_sound
                self._create_explosion(p['pos'], self.COLOR_PROJECTILE_CANNON, 5)
                
                if p['target']['health'] <= 0:
                    self.score += 10
                    self.step_reward += 1
                    self.money += 2
                    self._create_explosion(p['target']['pos'], self.COLOR_ENEMY, 20)
                    self.enemies.remove(p['target'])
                    # sfx: enemy_death_sound
                
                self.projectiles.remove(p)
            else:
                p['pos'][0] += (vec[0] / dist) * speed
                p['pos'][1] += (vec[1] / dist) * speed

    def _update_particles(self):
        for p in reversed(self.particles):
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _create_explosion(self, pos, color, count):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': random.randint(10, 20),
                'color': color,
            })

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
            self.step_reward -= 100 # Losing penalty
        if self.current_wave > self.TOTAL_WAVES and len(self.enemies) == 0:
            self.game_over = True
            self.step_reward += 100 # Winning bonus
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        
        return self.game_over

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "money": self.money,
            "base_health": self.base_health,
            "wave": self.current_wave,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_path()
        self._render_placement_grid()
        self._render_base()
        self._render_towers()
        self._render_enemies()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_iso_cube(self, surface, pos, size, color, stroke_color):
        x, y = pos
        w, h, z = size
        
        points = [
            (x, y - z),
            (x + w, y - z + h),
            (x, y - z + h * 2),
            (x - w, y - z + h),
        ]
        pygame.gfxdraw.aapolygon(surface, points, stroke_color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

        top_points = [
            (x, y - z),
            (x + w, y - z + h),
            (x, y - z + h * 2),
            (x - w, y - z + h)
        ]
        
        # Top face
        top_face_points = [
            (x, y - z),
            (x + w, y - z + h),
            (x, y - z + 2*h),
            (x - w, y - z + h)
        ]
        pygame.gfxdraw.filled_polygon(surface, top_face_points, tuple(min(255, c+20) for c in color))
        pygame.gfxdraw.aapolygon(surface, top_face_points, stroke_color)

        # Left face
        left_face_points = [
            (x-w, y-z+h),
            (x, y-z+2*h),
            (x, y+2*h),
            (x-w, y+h)
        ]
        pygame.gfxdraw.filled_polygon(surface, left_face_points, tuple(max(0, c-20) for c in color))
        
        # Right face
        right_face_points = [
            (x, y-z+2*h),
            (x+w, y-z+h),
            (x+w, y+h),
            (x, y+2*h)
        ]
        pygame.gfxdraw.filled_polygon(surface, right_face_points, tuple(max(0, c-40) for c in color))

    def _render_path(self):
        for x, y in self.path_cart:
            world_pos = self._cart_to_world(x, y)
            self._render_iso_cube(self.screen, world_pos, (self.TILE_WIDTH, self.TILE_HEIGHT, 0), self.COLOR_PATH, self.COLOR_GRID)

    def _render_placement_grid(self):
        for x, y in self.placement_grid:
            world_pos = self._cart_to_world(x, y)
            self._render_iso_cube(self.screen, world_pos, (self.TILE_WIDTH, self.TILE_HEIGHT, 0), self.COLOR_BG, self.COLOR_GRID)

    def _render_base(self):
        pos = self._cart_to_world(0,0)
        self._render_iso_cube(self.screen, pos, (self.TILE_WIDTH, self.TILE_HEIGHT, 20), self.COLOR_BASE, self.COLOR_BASE_STROKE)
        
        # Health bar
        bar_w = 60
        bar_h = 8
        bar_x = pos[0] - bar_w // 2
        bar_y = pos[1] - 40
        health_ratio = self.base_health / self.BASE_HEALTH_MAX
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, bar_w * health_ratio, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_w, bar_h), 1)

    def _render_towers(self):
        for tower in self.towers:
            tower_def = self.tower_definitions[tower['type']]
            height = 10 if tower['type'] == 0 else 5
            self._render_iso_cube(self.screen, tower['world_pos'], (self.TILE_WIDTH * 0.6, self.TILE_HEIGHT * 0.6, height), tower_def['color'], (50,50,50))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos_x, pos_y = int(enemy['pos'][0]), int(enemy['pos'][1])
            pygame.gfxdraw.filled_circle(self.screen, pos_x, pos_y, 6, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_x, pos_y, 6, self.COLOR_ENEMY_STROKE)

            # Health bar
            bar_w = 20
            bar_h = 4
            bar_x = pos_x - bar_w // 2
            bar_y = pos_y - 15
            health_ratio = max(0, enemy['health'] / enemy['max_health'])
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, bar_w * health_ratio, bar_h))

    def _render_projectiles(self):
        for p in self.projectiles:
            color = self.COLOR_PROJECTILE_CANNON if p['type'] == 0 else self.COLOR_PROJECTILE_GATLING
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, color)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 3, color)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 20))))
            color = p['color'] + (alpha,)
            size = int(max(1, 4 * (p['life'] / 20)))
            temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            self.screen.blit(temp_surf, (int(p['pos'][0] - size), int(p['pos'][1] - size)), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_cursor(self):
        grid_x, grid_y = self.cursor_pos
        cart_pos = (grid_x + self.grid_offset[0], grid_y + self.grid_offset[1])
        world_pos = self._cart_to_world(*cart_pos)
        
        is_valid = cart_pos in self.placement_grid and not any(t['cart_pos'] == cart_pos for t in self.towers)
        tower_def = self.tower_definitions[self.selected_tower_type]
        has_money = self.money >= tower_def['cost']
        
        color = self.COLOR_GRID_HOVER if is_valid and has_money else (255, 0, 0)
        
        temp_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        self._render_iso_cube(temp_surf, world_pos, (self.TILE_WIDTH, self.TILE_HEIGHT, 0), color + (50,), color + (150,))
        
        # Show range indicator
        pygame.gfxdraw.aacircle(temp_surf, int(world_pos[0]), int(world_pos[1]), tower_def['range'], color + (100,))
        self.screen.blit(temp_surf, (0,0))
        
    def _render_text(self, text, pos, font, color=COLOR_UI_TEXT, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        # Top-left: Score and Money
        self._render_text(f"SCORE: {self.score}", (10, 10), self.font_small)
        self._render_text(f"MONEY: ${self.money}", (10, 30), self.font_small)
        
        # Top-center: Wave Info
        if self.wave_countdown > 0 and self.current_wave < self.TOTAL_WAVES:
            secs = self.wave_countdown / self.FPS
            self._render_text(f"Wave {self.current_wave + 1} starting in {secs:.1f}s", (self.SCREEN_WIDTH // 2, 20), self.font_large, center=True)
        elif self.current_wave > 0:
            remaining = len(self.enemies) + len(self.enemies_to_spawn)
            self._render_text(f"WAVE {self.current_wave}/{self.TOTAL_WAVES} | ENEMIES: {remaining}", (self.SCREEN_WIDTH // 2, 20), self.font_large, center=True)

        # Bottom-right: Tower Selection
        ui_box = pygame.Rect(self.SCREEN_WIDTH - 210, self.SCREEN_HEIGHT - 80, 200, 70)
        pygame.draw.rect(self.screen, (0,0,0,150), ui_box, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, ui_box, 1, border_radius=5)
        
        tower_def = self.tower_definitions[self.selected_tower_type]
        self._render_text(f"Build: {tower_def['name']}", (ui_box.x + 10, ui_box.y + 5), self.font_small)
        self._render_text(f"Cost: ${tower_def['cost']}", (ui_box.x + 10, ui_box.y + 25), self.font_small, color=(0,255,0) if self.money >= tower_def['cost'] else (255,100,100))
        self._render_text(f"Dmg: {tower_def['damage']} | Rng: {tower_def['range']}", (ui_box.x + 10, ui_box.y + 45), self.font_small)

    def _render_game_over(self):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        won = self.current_wave > self.TOTAL_WAVES
        message = "VICTORY" if won else "GAME OVER"
        color = (0, 255, 150) if won else (255, 50, 50)
        
        self._render_text(message, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20), self.font_huge, color, center=True)
        self._render_text(f"Final Score: {self.score}", (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 40), self.font_large, center=True)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Manual Control Mapping ---
    # This is a basic mapping. A real human interface would be more complex.
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Create a window to display the game
    pygame.display.set_caption("Tower Defense")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = env.action_space.sample()
    action.fill(0)

    while not done:
        # Get human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        move_action = 0
        for key, move in key_to_action.items():
            if keys[key]:
                move_action = move
                break # Prioritize first key found
        
        action[0] = move_action
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
    pygame.quit()