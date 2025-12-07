
# Generated: 2025-08-28T03:13:19.785193
# Source Brief: brief_01957.md
# Brief Index: 1957

        
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
        "Controls: Arrow keys to move cursor. SHIFT to cycle tower type. SPACE to place selected tower."
    )

    game_description = (
        "Isometric tower defense. Place towers to defend your base from waves of enemies."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 20, 20
        self.TILE_W, self.TILE_H = 32, 16
        self.ISO_OFFSET_X = self.WIDTH // 2
        self.ISO_OFFSET_Y = 80

        # Colors
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (40, 55, 65)
        self.COLOR_PATH = (60, 80, 90)
        self.COLOR_PLACEABLE = (45, 65, 80, 100)
        self.COLOR_BASE = (0, 150, 200)
        self.COLOR_ENEMY = (220, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_UI_BG = (15, 20, 25, 200)
        
        self.TOWER_COLORS = [
            (240, 180, 50), # Short range
            (50, 200, 100), # Medium range
            (180, 100, 250) # Long range
        ]
        
        # Game constants
        self.MAX_STEPS = 3000 # Increased for longer games
        self.MAX_WAVES = 20
        self.INITIAL_RESOURCES = 100
        self.INITIAL_BASE_HEALTH = 1000
        self.INTER_WAVE_DELAY = 150 # 5 seconds at 30fps

        self.TOWER_SPECS = {
            0: {"cost": 10, "range": 2.5, "damage": 25, "cooldown": 20, "splash": 0, "projectile_speed": 8},
            1: {"cost": 15, "range": 4.0, "damage": 15, "cooldown": 30, "splash": 0, "projectile_speed": 10},
            2: {"cost": 20, "range": 5.5, "damage": 10, "cooldown": 45, "splash": 1.0, "projectile_speed": 6},
        }

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 14)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)

        self._define_level()
        
        # State variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = pygame.Vector2(0, 0)
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        self.wave_in_progress = False
        self.inter_wave_timer = 0
        self.spawn_queue = []
        self.spawn_timer = 0
        
        self.reset()
        self.validate_implementation()

    def _define_level(self):
        self.path_waypoints = [
            pygame.Vector2(-1, 9), pygame.Vector2(4, 9), pygame.Vector2(4, 4),
            pygame.Vector2(9, 4), pygame.Vector2(9, 14), pygame.Vector2(15, 14),
            pygame.Vector2(15, 8), pygame.Vector2(21, 8)
        ]
        self.base_pos = pygame.Vector2(19, 8)
        
        self.placement_grid = np.ones((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            for x in range(int(min(p1.x, p2.x)), int(max(p1.x, p2.x) + 1)):
                if 0 <= x < self.GRID_WIDTH:
                    self.placement_grid[x, int(p1.y)] = 0
            for y in range(int(min(p1.y, p2.y)), int(max(p1.y, p2.y) + 1)):
                if 0 <= y < self.GRID_HEIGHT:
                    self.placement_grid[int(p1.x), y] = 0
    
    def _grid_to_screen(self, x, y):
        screen_x = self.ISO_OFFSET_X + (x - y) * self.TILE_W / 2
        screen_y = self.ISO_OFFSET_Y + (x + y) * self.TILE_H / 2
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.cursor_pos = pygame.Vector2(self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_tower_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        self.wave_in_progress = False
        self.inter_wave_timer = self.INTER_WAVE_DELAY // 2
        self.spawn_queue = []
        self.spawn_timer = 0

        # Reset placement grid occupancy
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.placement_grid[x,y] != 0:
                    self.placement_grid[x,y] = 1

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.clock.tick(30)
        self.steps += 1
        
        step_reward = -0.01 # Time penalty

        # Handle player input
        step_reward += self._handle_input(action)
        
        # Update game logic
        self._update_wave_logic()
        step_reward += self._update_spawner()
        step_reward += self._update_enemies()
        step_reward += self._update_towers()
        step_reward += self._update_projectiles()
        self._update_particles()
        
        self.score += step_reward
        
        terminated = self._check_termination()
        
        if terminated:
            if self.base_health <= 0:
                step_reward -= 100
            elif self.wave_number > self.MAX_WAVES:
                step_reward += 100
        
        return self._get_observation(), step_reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cursor movement
        if movement == 1: self.cursor_pos.y -= 1
        elif movement == 2: self.cursor_pos.y += 1
        elif movement == 3: self.cursor_pos.x -= 1
        elif movement == 4: self.cursor_pos.x += 1
        self.cursor_pos.x = np.clip(self.cursor_pos.x, 0, self.GRID_WIDTH - 1)
        self.cursor_pos.y = np.clip(self.cursor_pos.y, 0, self.GRID_HEIGHT - 1)

        # Cycle tower type (on press)
        if shift_held and not self.last_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: ui_cycle

        # Place tower (on press)
        reward = 0
        if space_held and not self.last_space_held:
            cx, cy = int(self.cursor_pos.x), int(self.cursor_pos.y)
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.placement_grid[cx, cy] == 1 and self.resources >= spec["cost"]:
                self.resources -= spec["cost"]
                self.placement_grid[cx, cy] = 2 # Mark as occupied
                self.towers.append({
                    "pos": pygame.Vector2(cx, cy),
                    "type": self.selected_tower_type,
                    "cooldown_timer": 0,
                    "target": None
                })
                # sfx: place_tower
                self._create_particles(self._grid_to_screen(cx + 0.5, cy + 0.5), self.TOWER_COLORS[self.selected_tower_type], 15, 2)

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        return reward

    def _update_wave_logic(self):
        if not self.wave_in_progress and not self.spawn_queue and not self.enemies:
            self.inter_wave_timer -= 1
            if self.inter_wave_timer <= 0:
                self._start_next_wave()
                return 10 # Wave clear reward
        return 0

    def _start_next_wave(self):
        if self.wave_number >= self.MAX_WAVES: return
        self.wave_number += 1
        self.wave_in_progress = True
        # sfx: wave_start
        
        num_enemies = 5 + self.wave_number * 2
        base_health = 50 * (1 + self.wave_number * 0.1)
        base_speed = 0.03 * (1 + self.wave_number * 0.05)
        
        for i in range(num_enemies):
            self.spawn_queue.append({
                "health": base_health * (1 + self.np_random.uniform(-0.1, 0.1)),
                "speed": base_speed * (1 + self.np_random.uniform(-0.1, 0.1)),
                "value": 2 + int(self.wave_number / 5),
                "delay": i * 15 # Spawn delay between enemies
            })

    def _update_spawner(self):
        if self.spawn_queue:
            self.spawn_timer += 1
            if self.spawn_timer >= self.spawn_queue[0]["delay"]:
                enemy_data = self.spawn_queue.pop(0)
                self.enemies.append({
                    "pos": self.path_waypoints[0].copy(),
                    "health": enemy_data["health"],
                    "max_health": enemy_data["health"],
                    "speed": enemy_data["speed"],
                    "value": enemy_data["value"],
                    "path_index": 1
                })
                if not self.spawn_queue:
                    self.spawn_timer = 0
                    self.wave_in_progress = False # All enemies spawned for this wave
        return 0

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            if enemy["path_index"] >= len(self.path_waypoints):
                self.base_health -= enemy["health"]
                # sfx: base_hit
                self._create_particles(self._grid_to_screen(self.base_pos.x, self.base_pos.y), self.COLOR_ENEMY, 30, 5)
                self.enemies.remove(enemy)
                continue

            target_pos = self.path_waypoints[enemy["path_index"]]
            direction = (target_pos - enemy["pos"]).normalize() if (target_pos - enemy["pos"]).length() > 0 else pygame.Vector2(0,0)
            enemy["pos"] += direction * enemy["speed"]

            if (enemy["pos"] - target_pos).length_squared() < 0.1:
                enemy["path_index"] += 1
        return reward
    
    def _update_towers(self):
        for tower in self.towers:
            spec = self.TOWER_SPECS[tower["type"]]
            if tower["cooldown_timer"] > 0:
                tower["cooldown_timer"] -= 1
                continue

            # Find target (furthest along path in range)
            best_target = None
            max_dist = -1
            
            for enemy in self.enemies:
                dist_sq = (tower["pos"] - enemy["pos"]).length_squared()
                if dist_sq < spec["range"] ** 2:
                    path_dist = enemy["path_index"] + (enemy["pos"] - self.path_waypoints[enemy["path_index"]-1]).length()
                    if path_dist > max_dist:
                        max_dist = path_dist
                        best_target = enemy
            
            if best_target:
                tower["target"] = best_target
                tower["cooldown_timer"] = spec["cooldown"]
                self.projectiles.append({
                    "start_pos": tower["pos"] + pygame.Vector2(0.5, 0.5),
                    "pos": tower["pos"] + pygame.Vector2(0.5, 0.5),
                    "target": tower["target"],
                    "speed": spec["projectile_speed"],
                    "damage": spec["damage"],
                    "splash": spec["splash"],
                    "type": tower["type"]
                })
                # sfx: tower_shoot
        return 0

    def _update_projectiles(self):
        reward = 0
        for proj in self.projectiles[:]:
            target = proj["target"]
            if target not in self.enemies:
                self.projectiles.remove(proj)
                continue
            
            target_pos = target["pos"] + pygame.Vector2(0.5, 0.5)
            direction = (target_pos - proj["pos"]).normalize()
            proj["pos"] += direction * proj["speed"] / self.TILE_W # Scale speed to grid units
            
            if (proj["pos"] - target_pos).length_squared() < 0.25:
                # sfx: enemy_hit
                reward += self._deal_damage(proj["pos"], proj["damage"], proj["splash"])
                self.projectiles.remove(proj)
        return reward
    
    def _deal_damage(self, center, damage, splash_radius):
        reward = 0
        hit_color = self.COLOR_ENEMY
        self._create_particles(self._grid_to_screen(center.x, center.y), (255, 255, 255), 10, 1)

        if splash_radius > 0:
            self._create_particles(self._grid_to_screen(center.x, center.y), self.TOWER_COLORS[2], 20, 3, 0.5)

        for enemy in self.enemies[:]:
            dist_sq = (enemy["pos"] + pygame.Vector2(0.5, 0.5) - center).length_squared()
            dmg_to_deal = 0
            if dist_sq < 0.25: # Direct hit
                dmg_to_deal = damage
            elif splash_radius > 0 and dist_sq < splash_radius**2:
                dmg_to_deal = damage * (1 - (dist_sq / splash_radius**2)) # Falloff

            if dmg_to_deal > 0:
                enemy["health"] -= dmg_to_deal
                reward += 0.1 # Reward for any hit
                if enemy["health"] <= 0:
                    reward += 1.0 # Kill reward
                    self.resources += enemy["value"]
                    # sfx: enemy_die
                    self._create_particles(self._grid_to_screen(enemy["pos"].x, enemy["pos"].y), hit_color, 20, 3)
                    self.enemies.remove(enemy)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1
            p["size"] *= 0.95
            if p["lifespan"] <= 0 or p["size"] < 0.5:
                self.particles.remove(p)

    def _check_termination(self):
        if self.base_health <= 0 or self.steps >= self.MAX_STEPS or self.wave_number > self.MAX_WAVES:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        
        # Collect all dynamic objects to sort for isometric rendering
        render_queue = []
        for enemy in self.enemies:
            render_queue.append(("enemy", enemy))
        for tower in self.towers:
            render_queue.append(("tower", tower))
        
        # Sort by screen-y coordinate for correct occlusion
        render_queue.sort(key=lambda item: (item[1]["pos"].x + item[1]["pos"].y))

        for item_type, item_data in render_queue:
            if item_type == "enemy": self._render_enemy(item_data)
            elif item_type == "tower": self._render_tower(item_data)

        self._render_base()
        self._render_projectiles()
        self._render_particles()
        self._render_cursor()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Render placeable areas
        placeable_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.placement_grid[x, y] == 1:
                    p1 = self._grid_to_screen(x, y)
                    p2 = self._grid_to_screen(x + 1, y)
                    p3 = self._grid_to_screen(x + 1, y + 1)
                    p4 = self._grid_to_screen(x, y + 1)
                    pygame.gfxdraw.filled_polygon(placeable_surf, [p1, p2, p3, p4], self.COLOR_PLACEABLE)
        self.screen.blit(placeable_surf, (0,0))
        
        # Render path
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            if p1.x == p2.x: # Vertical path
                for y in range(int(min(p1.y, p2.y)), int(max(p1.y, p2.y)) + 1):
                    self._draw_iso_rect(p1.x, y, self.COLOR_PATH)
            else: # Horizontal path
                for x in range(int(min(p1.x, p2.x)), int(max(p1.x, p2.x)) + 1):
                    self._draw_iso_rect(x, p1.y, self.COLOR_PATH)

    def _draw_iso_rect(self, x, y, color):
        p1 = self._grid_to_screen(x, y)
        p2 = self._grid_to_screen(x + 1, y)
        p3 = self._grid_to_screen(x + 1, y + 1)
        p4 = self._grid_to_screen(x, y + 1)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], color)
        pygame.gfxdraw.aapolygon(self.screen, [p1, p2, p3, p4], self.COLOR_GRID)

    def _render_base(self):
        x, y = self.base_pos.x, self.base_pos.y
        p_center = self._grid_to_screen(x + 0.5, y + 0.5)
        p_top = self._grid_to_screen(x + 0.5, y - 0.5)
        height = 20
        
        c1 = tuple(max(0, c - 40) for c in self.COLOR_BASE)
        c2 = self.COLOR_BASE
        c3 = tuple(min(255, c + 40) for c in self.COLOR_BASE)
        
        # 3D-ish block
        pygame.draw.polygon(self.screen, c1, [p_center, self._grid_to_screen(x+1.5, y+0.5), self._grid_to_screen(x+1.5, y-0.5), p_top])
        pygame.draw.polygon(self.screen, c2, [p_center, self._grid_to_screen(x-0.5, y+0.5), self._grid_to_screen(x-0.5, y-0.5), p_top])
        pygame.draw.polygon(self.screen, c3, [p_top, self._grid_to_screen(x+0.5, y-1.5), self._grid_to_screen(x+1.5, y-0.5)])

        # Health bar
        bar_w, bar_h = 30, 5
        health_pct = max(0, self.base_health / self.INITIAL_BASE_HEALTH)
        fill_w = int(bar_w * health_pct)
        pygame.draw.rect(self.screen, (80,0,0), (p_center[0]-bar_w//2, p_center[1]-height-10, bar_w, bar_h))
        pygame.draw.rect(self.screen, (0,200,0), (p_center[0]-bar_w//2, p_center[1]-height-10, fill_w, bar_h))

    def _render_enemy(self, enemy):
        x, y = enemy["pos"].x, enemy["pos"].y
        sx, sy = self._grid_to_screen(x + 0.5, y + 0.5)
        radius = 5
        pygame.gfxdraw.filled_circle(self.screen, sx, sy, radius, self.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(self.screen, sx, sy, radius, tuple(max(0, c-50) for c in self.COLOR_ENEMY))
        
        # Health bar
        bar_w, bar_h = 16, 2
        health_pct = max(0, enemy["health"] / enemy["max_health"])
        fill_w = int(bar_w * health_pct)
        pygame.draw.rect(self.screen, (80,0,0), (sx-bar_w//2, sy-radius-5, bar_w, bar_h))
        pygame.draw.rect(self.screen, (0,200,0), (sx-bar_w//2, sy-radius-5, fill_w, bar_h))

    def _render_tower(self, tower):
        x, y = tower["pos"].x, tower["pos"].y
        sx, sy = self._grid_to_screen(x + 0.5, y + 0.5)
        spec = self.TOWER_SPECS[tower["type"]]
        color = self.TOWER_COLORS[tower["type"]]
        dark_color = tuple(max(0, c - 40) for c in color)
        
        # Base
        p1 = self._grid_to_screen(x + 0.2, y + 0.2)
        p2 = self._grid_to_screen(x + 0.8, y + 0.2)
        p3 = self._grid_to_screen(x + 0.8, y + 0.8)
        p4 = self._grid_to_screen(x + 0.2, y + 0.8)
        pygame.gfxdraw.filled_polygon(self.screen, [p1, p2, p3, p4], dark_color)
        
        # Top part based on type
        if tower["type"] == 0: # Square
            pygame.gfxdraw.box(self.screen, (sx-4, sy-10, 8, 8), color)
        elif tower["type"] == 1: # Triangle
            pygame.gfxdraw.filled_trigon(self.screen, sx, sy-12, sx-5, sy-5, sx+5, sy-5, color)
        elif tower["type"] == 2: # Circle
            pygame.gfxdraw.filled_circle(self.screen, sx, sy-8, 5, color)
            pygame.gfxdraw.aacircle(self.screen, sx, sy-8, 5, dark_color)

    def _render_projectiles(self):
        for proj in self.projectiles:
            sx, sy = self._grid_to_screen(proj["pos"].x, proj["pos"].y)
            color = self.TOWER_COLORS[proj["type"]]
            pygame.gfxdraw.filled_circle(self.screen, sx, sy, 3, color)

    def _render_particles(self):
        for p in self.particles:
            size = int(p["size"])
            if size > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), size, p["color"])

    def _render_cursor(self):
        cx, cy = int(self.cursor_pos.x), int(self.cursor_pos.y)
        spec = self.TOWER_SPECS[self.selected_tower_type]
        can_place = self.placement_grid[cx, cy] == 1 and self.resources >= spec["cost"]
        color = (0, 255, 0) if can_place else (255, 0, 0)
        
        # Cursor highlight
        p1 = self._grid_to_screen(cx, cy)
        p2 = self._grid_to_screen(cx + 1, cy)
        p3 = self._grid_to_screen(cx + 1, cy + 1)
        p4 = self._grid_to_screen(cx, cy + 1)
        pygame.draw.lines(self.screen, color, True, [p1, p2, p3, p4], 2)
        
        # Range indicator
        range_surf = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        center_sx, center_sy = self._grid_to_screen(cx + 0.5, cy + 0.5)
        radius_x = spec["range"] * self.TILE_W / 2
        radius_y = spec["range"] * self.TILE_H / 2
        if radius_x > 0 and radius_y > 0:
            rect = pygame.Rect(0, 0, int(radius_x * 2), int(radius_y * 2))
            rect.center = (center_sx, center_sy)
            pygame.draw.ellipse(range_surf, (*color, 60), rect)
            pygame.draw.ellipse(range_surf, (*color, 120), rect, 1)
        self.screen.blit(range_surf, (0,0))
        
    def _render_ui(self):
        # Top bar
        pygame.gfxdraw.box(self.screen, (0, 0, self.WIDTH, 40), self.COLOR_UI_BG)
        self._draw_text(f"❤️ {int(self.base_health)}", (20, 10), self.font_medium)
        self._draw_text(f"💰 {self.resources}", (180, 10), self.font_medium)
        self.score = int(self.score)
        self._draw_text(f"SCORE: {self.score}", (320, 10), self.font_medium)
        self._draw_text(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", (480, 10), self.font_medium)

        # Bottom bar
        pygame.gfxdraw.box(self.screen, (0, self.HEIGHT - 60, self.WIDTH, 60), self.COLOR_UI_BG)
        self._draw_text("TOWER SELECT (SHIFT)", (self.WIDTH//2, self.HEIGHT - 50), self.font_small, center=True)
        
        for i, spec in self.TOWER_SPECS.items():
            is_selected = i == self.selected_tower_type
            x_pos = self.WIDTH//2 + (i - 1) * 120
            
            box_color = self.TOWER_COLORS[i] if self.resources >= spec['cost'] else (100,100,100)
            if is_selected:
                pygame.draw.rect(self.screen, (255,255,255), (x_pos - 42, self.HEIGHT - 37, 84, 34), 2, 3)

            pygame.gfxdraw.box(self.screen, (x_pos - 40, self.HEIGHT - 35, 80, 30), box_color)
            self._draw_text(f"Cost: {spec['cost']}", (x_pos, self.HEIGHT - 25), self.font_small, center=True)
            
    def _draw_text(self, text, pos, font, color=None, center=False):
        color = color or self.COLOR_TEXT
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _create_particles(self, pos, color, count, max_speed, lifespan_mult=1.0):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, max_speed)
            vel = pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            self.particles.append({
                "pos": pygame.Vector2(pos),
                "vel": vel,
                "lifespan": self.np_random.integers(15, 30) * lifespan_mult,
                "size": self.np_random.uniform(2, 5),
                "color": color
            })
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "resources": self.resources,
            "base_health": self.base_health,
            "enemies_left": len(self.enemies) + len(self.spawn_queue)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    # Override screen to be the display
    env.screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Isometric Tower Defense")
    
    terminated = False
    total_reward = 0
    
    # Game loop
    while not terminated:
        # Pygame event handling
        movement = 0 # no-op
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # The _get_observation method already draws everything to env.screen
        # So we just need to flip the display
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            # Optional: Short pause before reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            terminated = False

    env.close()