
# Generated: 2025-08-28T06:03:11.601538
# Source Brief: brief_05770.md
# Brief Index: 5770

        
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


# Helper classes for game entities

class Tower:
    def __init__(self, pos, tower_type, iso_offset):
        self.pos = pos
        self.type = tower_type
        self.iso_offset = iso_offset
        self.fire_timer = 0
        self.target = None

        stats = TOWER_STATS[tower_type]
        self.range = stats["range"]
        self.damage = stats["damage"]
        self.cooldown = stats["cooldown"]
        self.color = stats["color"]
        self.projectile_speed = stats.get("projectile_speed", 0)
        self.aoe_radius = stats.get("aoe_radius", 0)

    def _iso_to_screen(self, x, y):
        screen_x = self.iso_offset[0] + (x - y) * TILE_WIDTH_HALF
        screen_y = self.iso_offset[1] + (x + y) * TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def find_target(self, enemies):
        # If current target is invalid or out of range, find a new one
        if self.target and (self.target.health <= 0 or self._distance_to(self.target) > self.range):
            self.target = None

        if not self.target:
            in_range_enemies = [e for e in enemies if self._distance_to(e) <= self.range]
            if in_range_enemies:
                # Target enemy closest to the base
                self.target = min(in_range_enemies, key=lambda e: e.remaining_dist)

    def _distance_to(self, enemy):
        return math.hypot(self.pos[0] - enemy.grid_pos[0], self.pos[1] - enemy.grid_pos[1])

    def update(self, enemies, projectiles, particles, np_random):
        self.fire_timer = max(0, self.fire_timer - 1)
        self.find_target(enemies)

        if self.target and self.fire_timer == 0:
            self.fire_timer = self.cooldown
            # sfx: tower_fire
            if self.type == "sniper":
                # Hitscan attack
                self.target.take_damage(self.damage)
                particles.append(Impact(self.target.pos, (255, 255, 100), 15, np_random))
            elif self.type == "splash":
                # Area of effect projectile
                projectiles.append(Projectile(self.pos, self.target, self.damage, self.projectile_speed, self.iso_offset, aoe_radius=self.aoe_radius))
            else: # basic
                projectiles.append(Projectile(self.pos, self.target, self.damage, self.projectile_speed, self.iso_offset))

    def draw(self, surface):
        screen_pos = self._iso_to_screen(*self.pos)
        base_points = [
            (screen_pos[0], screen_pos[1] - 10),
            (screen_pos[0] + 12, screen_pos[1] - 4),
            (screen_pos[0], screen_pos[1] + 2),
            (screen_pos[0] - 12, screen_pos[1] - 4),
        ]
        pygame.gfxdraw.filled_polygon(surface, base_points, self.color)
        pygame.gfxdraw.aapolygon(surface, base_points, self.color)

        if self.type == "sniper":
            top_points = [
                (screen_pos[0], screen_pos[1] - 25),
                (screen_pos[0] + 5, screen_pos[1] - 22),
                (screen_pos[0], screen_pos[1] - 19),
                (screen_pos[0] - 5, screen_pos[1] - 22),
            ]
            pygame.gfxdraw.filled_polygon(surface, top_points, (200, 200, 255))
            pygame.gfxdraw.aapolygon(surface, top_points, (200, 200, 255))
        elif self.type == "splash":
            pygame.gfxdraw.filled_circle(surface, screen_pos[0], screen_pos[1] - 15, 8, (255,165,0))
            pygame.gfxdraw.aacircle(surface, screen_pos[0], screen_pos[1] - 15, 8, (255,165,0))


class Enemy:
    def __init__(self, health, speed, path, iso_offset, np_random):
        self.max_health = health
        self.health = health
        self.speed = speed
        self.path = path
        self.path_index = 0
        self.pos = list(path[0])
        self.grid_pos = list(path[0])
        self.iso_offset = iso_offset
        self.np_random = np_random
        self.value = 10 # gold dropped on kill
        self.reached_base = False
        self.remaining_dist = self._calculate_remaining_dist()

    def _iso_to_screen(self, x, y):
        screen_x = self.iso_offset[0] + (x - y) * TILE_WIDTH_HALF
        screen_y = self.iso_offset[1] + (x + y) * TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)
    
    def _calculate_remaining_dist(self):
        dist = math.hypot(self.path[self.path_index+1][0] - self.pos[0], self.path[self.path_index+1][1] - self.pos[1])
        for i in range(self.path_index + 1, len(self.path) - 1):
            dist += math.hypot(self.path[i+1][0] - self.path[i][0], self.path[i+1][1] - self.path[i][1])
        return dist

    def update(self):
        if self.path_index >= len(self.path) - 1:
            self.reached_base = True
            return

        target_pos = self.path[self.path_index + 1]
        direction = (target_pos[0] - self.pos[0], target_pos[1] - self.pos[1])
        distance = math.hypot(*direction)

        if distance < self.speed:
            self.path_index += 1
            self.pos = list(target_pos)
        else:
            move = (direction[0] / distance * self.speed, direction[1] / distance * self.speed)
            self.pos[0] += move[0]
            self.pos[1] += move[1]
        
        self.grid_pos = (round(self.pos[0]), round(self.pos[1]))
        self.remaining_dist = self._calculate_remaining_dist()

    def take_damage(self, amount):
        self.health = max(0, self.health - amount)

    def draw(self, surface):
        screen_pos = self._iso_to_screen(*self.pos)
        size = 10
        points = [
            (screen_pos[0], screen_pos[1]),
            (screen_pos[0] + size, screen_pos[1] - size / 2),
            (screen_pos[0], screen_pos[1] - size),
            (screen_pos[0] - size, screen_pos[1] - size / 2),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, (255, 50, 50))
        pygame.gfxdraw.aapolygon(surface, points, (255, 150, 150))
        
        # Health bar
        bar_width = 24
        bar_height = 4
        health_pct = self.health / self.max_health
        health_bar_pos = (screen_pos[0] - bar_width // 2, screen_pos[1] - 25)
        pygame.draw.rect(surface, (50, 50, 50), (*health_bar_pos, bar_width, bar_height))
        pygame.draw.rect(surface, (50, 255, 50), (*health_bar_pos, int(bar_width * health_pct), bar_height))

class Projectile:
    def __init__(self, start_pos, target, damage, speed, iso_offset, aoe_radius=0):
        self.start_pos_grid = start_pos
        self.target = target
        self.damage = damage
        self.speed = speed
        self.iso_offset = iso_offset
        self.aoe_radius = aoe_radius
        self.hit = False
        
        self.pos = self._iso_to_screen(*self.start_pos_grid)
        self.pos = [self.pos[0], self.pos[1] - 15] # Start from tower's top

    def _iso_to_screen(self, x, y):
        screen_x = self.iso_offset[0] + (x - y) * TILE_WIDTH_HALF
        screen_y = self.iso_offset[1] + (x + y) * TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def update(self):
        if self.hit or self.target.health <= 0:
            self.hit = True # Mark for deletion
            return

        target_screen_pos = self._iso_to_screen(*self.target.pos)
        direction = (target_screen_pos[0] - self.pos[0], target_screen_pos[1] - 10 - self.pos[1])
        distance = math.hypot(*direction)

        if distance < self.speed:
            self.hit = True
        else:
            self.pos[0] += direction[0] / distance * self.speed
            self.pos[1] += direction[1] / distance * self.speed

    def draw(self, surface):
        color = (100, 200, 255) if self.aoe_radius == 0 else (255, 200, 100)
        pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), 3, color)
        pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), 3, color)

class Particle:
    def __init__(self, pos, color, max_age, np_random):
        self.pos = list(pos)
        self.color = color
        self.max_age = max_age
        self.age = 0
        self.np_random = np_random
        self.vel = [self.np_random.uniform(-2, 2), self.np_random.uniform(-3, -1)]

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[1] += 0.1 # Gravity
        self.age += 1
        return self.age >= self.max_age

    def draw(self, surface):
        life_pct = 1 - (self.age / self.max_age)
        radius = int(3 * life_pct)
        if radius > 0:
            alpha = int(255 * life_pct)
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*self.color, alpha))
            surface.blit(temp_surf, (int(self.pos[0]) - radius, int(self.pos[1]) - radius))

class Impact(Particle):
     def __init__(self, pos, color, max_age, np_random):
        super().__init__(pos, color, max_age, np_random)
        self.vel = [self.np_random.uniform(-3, 3), self.np_random.uniform(-3, 3)]
        self.vel[1] += 0 # No gravity for impact sparks

class Explosion(Particle):
    def __init__(self, pos, radius, max_age, np_random):
        super().__init__(pos, (255, 165, 0), max_age, np_random)
        self.radius = radius
        self.current_radius = 0

    def update(self):
        self.age += 1
        self.current_radius = self.radius * math.sin((self.age / self.max_age) * math.pi / 2)
        return self.age >= self.max_age

    def draw(self, surface):
        alpha = int(150 * (1 - (self.age / self.max_age)))
        if self.current_radius > 0:
            pygame.gfxdraw.aacircle(surface, int(self.pos[0]), int(self.pos[1]), int(self.current_radius), (*self.color, alpha))
            pygame.gfxdraw.filled_circle(surface, int(self.pos[0]), int(self.pos[1]), int(self.current_radius), (*self.color, alpha // 2))


# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 24, 12
ISO_OFFSET = (SCREEN_WIDTH // 2, 80)

TOWER_STATS = {
    "basic": {"cost": 100, "damage": 10, "range": 4, "cooldown": 30, "color": (0, 200, 200), "projectile_speed": 8},
    "sniper": {"cost": 250, "damage": 50, "range": 8, "cooldown": 90, "color": (150, 150, 255)},
    "splash": {"cost": 175, "damage": 15, "range": 3, "cooldown": 60, "color": (255, 120, 0), "projectile_speed": 5, "aoe_radius": 1.5},
}
TOWER_TYPES = list(TOWER_STATS.keys())


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Use arrows to move cursor. SHIFT to cycle tower types. SPACE to build on a green tile or to start the wave."
    )

    game_description = (
        "Defend your base from waves of enemies by strategically placing various towers on the isometric grid."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Colors
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_PATH = (50, 60, 70)
        self.COLOR_GRID = (70, 80, 90)
        self.COLOR_GRID_HOVER = (150, 255, 150)
        self.COLOR_GRID_INVALID = (255, 150, 150)
        self.COLOR_BASE = (50, 200, 50)
        
        self._init_map()
        self.reset()
        self.validate_implementation()
    
    def _init_map(self):
        self.path_waypoints = [
            (-2, 4), (4, 4), (4, 10), (10, 10), (10, 16), (16, 16), (16, 10), (22, 10)
        ]
        
        build_tiles_raw = [
            (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
            (2, 6), (3, 6), (5, 6), (6, 6),
            (2, 8), (3, 8), (5, 8), (6, 8),
            (8, 8), (9, 8), (11, 8), (12, 8),
            (8, 12), (9, 12), (11, 12), (12, 12),
            (14, 12), (15, 12), (17, 12), (18, 12),
            (14, 14), (15, 14), (17, 14), (18, 14),
        ]
        self.buildable_tiles = {pos for pos in build_tiles_raw}
        self.base_pos = (20, 10)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.base_health = 100
        self.gold = 300
        self.wave_number = 0
        self.game_phase = "interwave" # interwave, wave_spawning, wave_active
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = (2, 2)
        self.selected_tower_idx = 0
        
        self._wave_spawn_timer = 0
        self._wave_enemies_to_spawn = 0

        self.last_action_was_press = {'space': False, 'shift': False}
        
        return self._get_observation(), self._get_info()

    def _iso_to_screen(self, x, y):
        screen_x = ISO_OFFSET[0] + (x - y) * TILE_WIDTH_HALF
        screen_y = ISO_OFFSET[1] + (x + y) * TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _start_next_wave(self):
        self.game_phase = "wave_spawning"
        self.wave_number += 1
        num_enemies = 3 + self.wave_number * 2
        self._wave_enemies_to_spawn = num_enemies
        self._wave_spawn_timer = 0
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Action Handling ---
        # Handle button presses (single trigger per press)
        space_press = space_held and not self.last_action_was_press['space']
        shift_press = shift_held and not self.last_action_was_press['shift']
        self.last_action_was_press['space'] = space_held
        self.last_action_was_press['shift'] = shift_held
        
        if self.game_phase == "interwave":
            # Move cursor
            dx, dy = 0, 0
            if movement == 1: dy = -1 # Up
            elif movement == 2: dy = 1 # Down
            elif movement == 3: dx = -1 # Left
            elif movement == 4: dx = 1 # Right
            
            if dx != 0 or dy != 0:
                potential_pos = (self.cursor_pos[0] + dx, self.cursor_pos[1] + dy)
                if potential_pos in self.buildable_tiles:
                    self.cursor_pos = potential_pos
            
            # Cycle tower type
            if shift_press:
                self.selected_tower_idx = (self.selected_tower_idx + 1) % len(TOWER_TYPES)
                # sfx: ui_blip

            # Place tower or start wave
            if space_press:
                can_build = self.cursor_pos in self.buildable_tiles and \
                            self.cursor_pos not in [t.pos for t in self.towers]
                
                selected_type = TOWER_TYPES[self.selected_tower_idx]
                cost = TOWER_STATS[selected_type]["cost"]

                if can_build and self.gold >= cost:
                    self.gold -= cost
                    self.towers.append(Tower(self.cursor_pos, selected_type, ISO_OFFSET))
                    # sfx: build_tower
                else:
                    self._start_next_wave()
                    # sfx: wave_start
        
        # --- Game Logic Update ---
        if self.game_phase == "wave_spawning":
            self._wave_spawn_timer -= 1
            if self._wave_spawn_timer <= 0 and self._wave_enemies_to_spawn > 0:
                self._wave_spawn_timer = 30 # Spawn every second
                self._wave_enemies_to_spawn -= 1
                
                health = 20 * (1.1 ** self.wave_number)
                speed = 0.05 * (1.05 ** self.wave_number)
                self.enemies.append(Enemy(health, speed, self.path_waypoints, ISO_OFFSET, self.np_random))

            if self._wave_enemies_to_spawn == 0 and not self.enemies:
                self.game_phase = "interwave" # Spawned all, but they were killed instantly
                reward += 1 # Wave complete reward
                self.gold += 100 + self.wave_number * 10
        
        if self.game_phase in ["wave_spawning", "wave_active"]:
            # Update towers
            for tower in self.towers:
                tower.update(self.enemies, self.projectiles, self.particles, self.np_random)

            # Update projectiles
            for p in self.projectiles:
                p.update()
                if p.hit:
                    # sfx: projectile_hit
                    if p.aoe_radius > 0:
                        self.particles.append(Explosion(p.pos, p.aoe_radius * TILE_WIDTH_HALF, 20, self.np_random))
                        for enemy in self.enemies:
                            dist = math.hypot(p.target.pos[0] - enemy.pos[0], p.target.pos[1] - enemy.pos[1])
                            if dist <= p.aoe_radius:
                                enemy.take_damage(p.damage)
                    else:
                        p.target.take_damage(p.damage)
                        self.particles.append(Impact(p.pos, (100,200,255), 10, self.np_random))

            self.projectiles = [p for p in self.projectiles if not p.hit]

            # Update enemies
            enemies_survived = []
            for enemy in self.enemies:
                enemy.update()
                if enemy.reached_base:
                    self.base_health -= 10
                    reward -= 1.0 # -0.1 per health * 10
                    # sfx: base_damage
                elif enemy.health <= 0:
                    reward += 0.1
                    self.gold += enemy.value
                    self.score += 10
                    # sfx: enemy_die
                    for _ in range(5):
                        self.particles.append(Particle(self._iso_to_screen(*enemy.pos), (255, 80, 80), 30, self.np_random))
                else:
                    enemies_survived.append(enemy)
            self.enemies = enemies_survived

            # Check for end of wave
            if self.game_phase == "wave_spawning" and self._wave_enemies_to_spawn == 0:
                self.game_phase = "wave_active"
            
            if self.game_phase == "wave_active" and not self.enemies:
                self.game_phase = "interwave"
                reward += 1.0 # Wave complete reward
                self.gold += 100 + self.wave_number * 10
                # sfx: wave_complete
        
        # Update particles
        self.particles = [p for p in self.particles if not p.update()]
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.wave_number > 20:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Lose penalty
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if self.base_health <= 0:
            return True
        if self.wave_number > 20: # Survived 20 waves
            return True
        if self.steps >= 5000: # Max steps
             return True
        return False
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "gold": self.gold,
            "base_health": self.base_health,
            "wave": self.wave_number,
            "phase": self.game_phase
        }

    def _render_game(self):
        # Draw path
        for i in range(len(self.path_waypoints) - 1):
            p1 = self._iso_to_screen(*self.path_waypoints[i])
            p2 = self._iso_to_screen(*self.path_waypoints[i+1])
            pygame.draw.line(self.screen, self.COLOR_PATH, p1, p2, TILE_HEIGHT_HALF * 2 + 4)
        
        # Draw buildable tiles
        for tile_pos in self.buildable_tiles:
            screen_pos = self._iso_to_screen(*tile_pos)
            points = [
                (screen_pos[0], screen_pos[1] - TILE_HEIGHT_HALF),
                (screen_pos[0] + TILE_WIDTH_HALF, screen_pos[1]),
                (screen_pos[0], screen_pos[1] + TILE_HEIGHT_HALF),
                (screen_pos[0] - TILE_WIDTH_HALF, screen_pos[1]),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # Draw base
        base_screen_pos = self._iso_to_screen(*self.base_pos)
        base_points = [
            (base_screen_pos[0], base_screen_pos[1] - 20),
            (base_screen_pos[0] + 20, base_screen_pos[1] - 10),
            (base_screen_pos[0], base_screen_pos[1]),
            (base_screen_pos[0] - 20, base_screen_pos[1] - 10),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, base_points, self.COLOR_BASE)
        pygame.gfxdraw.aapolygon(self.screen, base_points, self.COLOR_BASE)

        # Draw towers
        for tower in self.towers:
            tower.draw(self.screen)

        # Draw enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)

        # Draw projectiles
        for p in self.projectiles:
            p.draw(self.screen)
        
        # Draw particles
        for p in self.particles:
            p.draw(self.screen)
        
        # Draw cursor
        if self.game_phase == "interwave":
            occupied_tiles = {t.pos for t in self.towers}
            is_occupied = self.cursor_pos in occupied_tiles
            
            selected_type = TOWER_TYPES[self.selected_tower_idx]
            cost = TOWER_STATS[selected_type]["cost"]
            can_afford = self.gold >= cost
            
            cursor_color = self.COLOR_GRID_HOVER
            if is_occupied or not can_afford:
                cursor_color = self.COLOR_GRID_INVALID

            screen_pos = self._iso_to_screen(*self.cursor_pos)
            points = [
                (screen_pos[0], screen_pos[1] - TILE_HEIGHT_HALF),
                (screen_pos[0] + TILE_WIDTH_HALF, screen_pos[1]),
                (screen_pos[0], screen_pos[1] + TILE_HEIGHT_HALF),
                (screen_pos[0] - TILE_WIDTH_HALF, screen_pos[1]),
            ]
            pygame.draw.polygon(self.screen, cursor_color, points, 3)

            # Draw tower range preview
            if not is_occupied:
                tower_range = TOWER_STATS[selected_type]["range"]
                range_px = tower_range * TILE_WIDTH_HALF * 1.414 # Approximation
                s = pygame.Surface((range_px*2, range_px*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*cursor_color, 50), (range_px, range_px), range_px)
                self.screen.blit(s, (screen_pos[0] - range_px, screen_pos[1] - range_px))

    def _render_ui(self):
        # Top Bar
        ui_bar = pygame.Surface((SCREEN_WIDTH, 40))
        ui_bar.fill((20, 25, 30))
        
        health_text = self.font_small.render(f"Base Health: {max(0, self.base_health)}/100", True, self.COLOR_BASE)
        gold_text = self.font_small.render(f"Gold: {self.gold}", True, (255, 215, 0))
        wave_text = self.font_small.render(f"Wave: {self.wave_number}/20", True, (200, 200, 255))
        
        ui_bar.blit(health_text, (10, 10))
        ui_bar.blit(gold_text, (200, 10))
        ui_bar.blit(wave_text, (350, 10))
        
        self.screen.blit(ui_bar, (0, 0))

        # Bottom Bar (Tower Selection)
        bottom_bar = pygame.Surface((SCREEN_WIDTH, 60))
        bottom_bar.fill((20, 25, 30))
        
        if self.game_phase == "interwave":
            selected_type = TOWER_TYPES[self.selected_tower_idx]
            stats = TOWER_STATS[selected_type]
            
            tower_info = f"Build: {selected_type.upper()} | Cost: {stats['cost']} | Dmg: {stats['damage']} | Range: {stats['range']}"
            info_text = self.font_small.render(tower_info, True, (255, 255, 255))
            bottom_bar.blit(info_text, (10, 20))
            
            help_text = self.font_small.render("[SPACE] to build/start wave | [SHIFT] to cycle", True, (150, 150, 150))
            bottom_bar.blit(help_text, (SCREEN_WIDTH - help_text.get_width() - 10, 20))
        elif self.game_phase != "interwave":
            status_text = self.font_small.render("WAVE IN PROGRESS...", True, (255, 80, 80))
            bottom_bar.blit(status_text, (SCREEN_WIDTH // 2 - status_text.get_width() // 2, 20))

        self.screen.blit(bottom_bar, (0, SCREEN_HEIGHT - 60))

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.wave_number > 20:
                msg = "VICTORY!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
            
            text = self.font_large.render(msg, True, color)
            text_rect = text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            overlay.blit(text, text_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def close(self):
        pygame.quit()

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

# Example of how to run the environment
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Manual Play ---
    # Set to True to play manually with the keyboard
    MANUAL_PLAY = True
    
    if MANUAL_PLAY:
        # Re-initialize pygame to create a display window
        pygame.display.init()
        pygame.display.set_caption("Isometric Tower Defense")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        terminated = False
        while not terminated:
            # Map keyboard keys to the MultiDiscrete action space
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # For manual play, we need to decide when to step.
            # Since auto_advance is False, we step on any key press or hold.
            # A simple timer can make the wave phase feel real-time.
            
            # Poll for events
            should_step = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                if event.type == pygame.KEYDOWN:
                    should_step = True
            
            # In wave phase, we want to advance frames automatically
            if env.game_phase != 'interwave':
                should_step = True
                # Use a no-op action if no key is pressed
                if not any(keys):
                    action = [0, 0, 0]

            if should_step and not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
            
            # Render the observation to the display
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            # Cap the frame rate during waves for smoother visuals
            if env.game_phase != 'interwave':
                env.clock.tick(30)
                
        env.close()
    else:
        # --- Random Agent ---
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                print(f"Game Over. Final Info: {info}")
                obs, info = env.reset()
        env.close()