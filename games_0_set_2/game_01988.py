
# Generated: 2025-08-27T18:54:11.679552
# Source Brief: brief_01988.md
# Brief Index: 1988

        
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

    user_guide = (
        "Controls: Arrows to move cursor. Space to place selected tower. Shift to cycle tower types."
    )

    game_description = (
        "Defend your base from waves of invaders by strategically placing defensive towers in an isometric world."
    )

    auto_advance = True

    # --- Colors and Style ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID = (40, 50, 60)
    COLOR_PATH = (60, 75, 90)
    COLOR_BASE = (0, 150, 200)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_CURSOR = (255, 255, 0, 180)
    COLOR_TEXT = (220, 220, 230)
    COLOR_UI_BG = (50, 60, 70, 200)
    COLOR_HEALTH_GREEN = (40, 200, 40)
    COLOR_HEALTH_RED = (200, 40, 40)

    TOWER_SPECS = {
        0: {"name": "Cannon", "cost": 100, "range": 100, "damage": 25, "fire_rate": 45, "color": (200, 200, 200)}, # Slower, high damage
        1: {"name": "Turret", "cost": 75, "range": 120, "damage": 10, "fire_rate": 20, "color": (150, 150, 255)}  # Faster, low damage
    }

    WAVE_CONFIG = [
        {"count": 10, "health": 100, "speed": 1.0, "reward": 10},
        {"count": 15, "health": 120, "speed": 1.1, "reward": 15},
        {"count": 20, "health": 150, "speed": 1.2, "reward": 20},
        {"count": 25, "health": 180, "speed": 1.3, "reward": 25},
        {"count": 30, "health": 220, "speed": 1.5, "reward": 30}
    ]
    MAX_WAVES = len(WAVE_CONFIG)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 50)

        # --- World and Grid ---
        self.GRID_W, self.GRID_H = 20, 14
        self.TILE_W, self.TILE_H = 32, 16
        self.ISO_OFFSET_X = 320
        self.ISO_OFFSET_Y = 80
        
        self.path_coords = [
            (0, 5), (1, 5), (2, 5), (3, 5), (3, 4), (3, 3), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
            (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (8, 7), (9, 7), (10, 7), (11, 7), (12, 7),
            (12, 8), (12, 9), (12, 10), (13, 10), (14, 10), (15, 10), (16, 10), (17, 10), (18, 10),
            (19, 10)
        ]
        self.path_pixels = [self._grid_to_iso(x, y) for x, y in self.path_coords]
        self.base_pos_grid = self.path_coords[-1]
        self.base_pos_iso = self.path_pixels[-1]
        
        self.buildable_tiles = self._get_buildable_tiles()

        # Initialize state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.base_health = 0
        self.max_base_health = 0
        self.resources = 0
        self.wave_number = 0
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        self.cursor_pos = [0, 0]
        self.selected_tower_type = 0
        self.wave_spawn_timer = 0
        self.enemy_spawn_timer = 0
        self.enemies_left_to_spawn = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_base_hit_time = -1000

        self.reset()
        self.validate_implementation()

    def _get_buildable_tiles(self):
        path_set = set(self.path_coords)
        buildable = []
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                if (x, y) not in path_set:
                    buildable.append((x, y))
        return buildable

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.max_base_health = 100
        self.base_health = self.max_base_health
        self.resources = 250
        
        self.wave_number = 0
        self.wave_spawn_timer = 150 # Time before first wave
        self.enemies_left_to_spawn = 0
        self.enemy_spawn_timer = 0
        
        self.enemies.clear()
        self.towers.clear()
        self.projectiles.clear()
        self.particles.clear()
        
        self.cursor_pos = [self.GRID_W // 2, self.GRID_H // 2]
        self.selected_tower_type = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_base_hit_time = -self.steps

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # --- Handle player input ---
        self._handle_input(movement, space_held, shift_held)

        # --- Update game state ---
        self._update_waves()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()
        
        self.score += reward
        self.steps += 1
        
        # --- Check for termination ---
        terminated = False
        win = self.wave_number > self.MAX_WAVES and not self.enemies
        lose = self.base_health <= 0
        timeout = self.steps >= 3000

        if win:
            reward += 100
            self.game_over = True
            terminated = True
        elif lose:
            reward -= 100
            self.game_over = True
            terminated = True
        elif timeout:
            self.game_over = True
            terminated = True

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # Cycle tower type
        if shift_held and not self.prev_shift_held:
            self.selected_tower_type = (self.selected_tower_type + 1) % len(self.TOWER_SPECS)
            # sfx: ui_switch

        # Place tower
        if space_held and not self.prev_space_held:
            spec = self.TOWER_SPECS[self.selected_tower_type]
            if self.resources >= spec["cost"] and tuple(self.cursor_pos) in self.buildable_tiles:
                is_occupied = any(t.grid_pos == tuple(self.cursor_pos) for t in self.towers)
                if not is_occupied:
                    self.resources -= spec["cost"]
                    new_tower = Tower(tuple(self.cursor_pos), self.selected_tower_type, self)
                    self.towers.append(new_tower)
                    # sfx: place_tower
                    self._create_particles(new_tower.pos, (200, 200, 255), 15)

    def _update_waves(self):
        is_wave_active = self.enemies_left_to_spawn > 0 or self.enemies
        
        if not is_wave_active and self.wave_number <= self.MAX_WAVES:
            self.wave_spawn_timer -= 1
            if self.wave_spawn_timer <= 0:
                self.wave_number += 1
                if self.wave_number <= self.MAX_WAVES:
                    wave_data = self.WAVE_CONFIG[self.wave_number - 1]
                    self.enemies_left_to_spawn = wave_data["count"]
                    self.enemy_spawn_timer = 0
                    # sfx: wave_start

        if self.enemies_left_to_spawn > 0:
            self.enemy_spawn_timer -= 1
            if self.enemy_spawn_timer <= 0:
                self.enemy_spawn_timer = 30 # Time between enemies
                self.enemies_left_to_spawn -= 1
                wave_data = self.WAVE_CONFIG[self.wave_number - 1]
                self.enemies.append(Enemy(wave_data, self))
                # sfx: enemy_spawn
    
    def _update_towers(self):
        for tower in self.towers:
            tower.update(self.enemies)
        return 0

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p.update()
            if p.hit:
                reward += 0.1 # Reward for damaging
                p.target.health -= p.damage
                self._create_particles(p.pos, p.color, 5, 2)
                # sfx: projectile_hit
                if p.target.health <= 0:
                    reward += p.target.kill_reward
                    self.resources += p.target.kill_reward
                    self._create_particles(p.target.pos, (255, 80, 80), 20, 3)
                    self.enemies.remove(p.target)
                    # sfx: enemy_death
                self.projectiles.remove(p)
            elif p.is_dead():
                self.projectiles.remove(p)
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            enemy.update()
            if enemy.reached_base:
                damage = 10
                self.base_health -= damage
                reward -= damage * 0.1 # Penalty for base damage
                self.enemies.remove(enemy)
                self.last_base_hit_time = self.steps
                self._create_particles(self.base_pos_iso, (255, 50, 50), 30, 4)
                # sfx: base_damage
        self.base_health = max(0, self.base_health)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.lifespan <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid and path
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                p1 = self._grid_to_iso(x, y)
                p2 = self._grid_to_iso(x + 1, y)
                p3 = self._grid_to_iso(x, y + 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
                pygame.draw.line(self.screen, self.COLOR_GRID, p1, p3, 1)
        
        if len(self.path_pixels) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_pixels, 8)

        # Draw Base
        base_flash = self.steps - self.last_base_hit_time < 15
        base_color = self.COLOR_BASE_DMG if base_flash else self.COLOR_BASE
        base_rect = pygame.Rect(0, 0, 40, 30)
        base_rect.center = self.base_pos_iso
        pygame.draw.rect(self.screen, base_color, base_rect, border_radius=4)
        pygame.draw.rect(self.screen, (255,255,255), base_rect, 2, border_radius=4)

        # Draw Towers
        for tower in self.towers:
            tower.draw(self.screen)

        # Draw Enemies
        for enemy in sorted(self.enemies, key=lambda e: e.pos[1]):
            enemy.draw(self.screen)
        
        # Draw Projectiles
        for p in self.projectiles:
            p.draw(self.screen)

        # Draw Particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw Cursor
        cursor_iso = self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1])
        spec = self.TOWER_SPECS[self.selected_tower_type]
        is_placeable = (self.resources >= spec["cost"] and 
                        tuple(self.cursor_pos) in self.buildable_tiles and
                        not any(t.grid_pos == tuple(self.cursor_pos) for t in self.towers))

        # Cursor tile
        cursor_poly = [
            self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1]),
            self._grid_to_iso(self.cursor_pos[0] + 1, self.cursor_pos[1]),
            self._grid_to_iso(self.cursor_pos[0] + 1, self.cursor_pos[1] + 1),
            self._grid_to_iso(self.cursor_pos[0], self.cursor_pos[1] + 1),
        ]
        cursor_color = (0, 255, 0, 100) if is_placeable else (255, 0, 0, 100)
        pygame.gfxdraw.filled_polygon(self.screen, cursor_poly, cursor_color)
        pygame.gfxdraw.aapolygon(self.screen, cursor_poly, (255, 255, 255, 150))
        
        # Tower range indicator
        pygame.gfxdraw.aacircle(self.screen, cursor_iso[0], cursor_iso[1], spec["range"], (255, 255, 255, 50))


    def _render_ui(self):
        # UI Panel
        ui_panel = pygame.Surface((640, 70), pygame.SRCALPHA)
        ui_panel.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_panel, (0, 330))
        pygame.draw.line(self.screen, (100, 120, 140), (0, 330), (640, 330))

        # Resources
        self._draw_text(f"Gold: {self.resources}", (20, 20), self.font_medium)
        
        # Base Health Bar
        health_pct = self.base_health / self.max_base_health
        health_bar_w = 200
        health_bar_rect = pygame.Rect(620 - health_bar_w, 20, health_bar_w, 20)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, health_bar_rect, border_radius=4)
        fill_rect = health_bar_rect.copy()
        fill_rect.width = int(health_bar_w * health_pct)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, fill_rect, border_radius=4)
        self._draw_text("Base", (health_bar_rect.left - 45, 22), self.font_small)

        # Wave Info
        wave_text = f"Wave: {self.wave_number}/{self.MAX_WAVES}"
        if self.wave_spawn_timer > 0 and self.wave_number < self.MAX_WAVES:
            wave_text += f" (Next in {self.wave_spawn_timer // 30 + 1}s)"
        self._draw_text(wave_text, (250, 22), self.font_medium)
        
        # Selected Tower UI
        spec = self.TOWER_SPECS[self.selected_tower_type]
        self._draw_text(f"Selected: {spec['name']}", (20, 350), self.font_medium)
        self._draw_text(f"Cost: {spec['cost']}", (20, 375), self.font_small)
        self._draw_text(f"Dmg: {spec['damage']}", (120, 375), self.font_small)
        self._draw_text(f"Rng: {spec['range']}", (200, 375), self.font_small)
        self._draw_text(f"Rate: {60 / spec['fire_rate']:.1f}/s", (280, 375), self.font_small)
        
    def _render_game_over(self):
        overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        win = self.wave_number > self.MAX_WAVES and not self.enemies
        message = "VICTORY" if win else "DEFEAT"
        color = (100, 255, 100) if win else (255, 100, 100)
        
        self._draw_text(message, (320, 180), self.font_large, color, center=True)
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "resources": self.resources,
            "wave": self.wave_number
        }

    def _grid_to_iso(self, grid_x, grid_y):
        iso_x = self.ISO_OFFSET_X + (grid_x - grid_y) * (self.TILE_W / 2)
        iso_y = self.ISO_OFFSET_Y + (grid_x + grid_y) * (self.TILE_H / 2)
        return int(iso_x), int(iso_y)

    def _draw_text(self, text, pos, font, color=COLOR_TEXT, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)
        
    def _create_particles(self, pos, color, count, speed_mult=1.0):
        for _ in range(count):
            self.particles.append(Particle(pos, color, speed_mult))
    
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

class Enemy:
    def __init__(self, wave_data, env):
        self.env = env
        self.path_index = 0
        self.pos = list(self.env.path_pixels[0])
        self.max_health = wave_data["health"]
        self.health = self.max_health
        self.speed = wave_data["speed"]
        self.kill_reward = wave_data["reward"]
        self.reached_base = False
        self.size = 8

    def update(self):
        if self.path_index < len(self.env.path_pixels) - 1:
            target_pos = self.env.path_pixels[self.path_index + 1]
            direction = (target_pos[0] - self.pos[0], target_pos[1] - self.pos[1])
            dist = math.hypot(*direction)
            
            if dist < self.speed:
                self.path_index += 1
                self.pos = list(self.env.path_pixels[self.path_index])
            else:
                norm_dir = (direction[0] / dist, direction[1] / dist)
                self.pos[0] += norm_dir[0] * self.speed
                self.pos[1] += norm_dir[1] * self.speed
        else:
            self.reached_base = True

    def draw(self, screen):
        # Body
        draw_pos = (int(self.pos[0]), int(self.pos[1]))
        pygame.gfxdraw.filled_circle(screen, draw_pos[0], draw_pos[1], self.size, (200, 50, 50))
        pygame.gfxdraw.aacircle(screen, draw_pos[0], draw_pos[1], self.size, (255, 100, 100))
        
        # Health bar
        bar_w = 20
        bar_h = 4
        bar_x = draw_pos[0] - bar_w / 2
        bar_y = draw_pos[1] - self.size - bar_h - 2
        health_pct = self.health / self.max_health
        
        pygame.draw.rect(screen, GameEnv.COLOR_HEALTH_RED, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(screen, GameEnv.COLOR_HEALTH_GREEN, (bar_x, bar_y, int(bar_w * health_pct), bar_h))

class Tower:
    def __init__(self, grid_pos, type_id, env):
        self.env = env
        self.grid_pos = grid_pos
        self.pos = env._grid_to_iso(*grid_pos)
        self.type_id = type_id
        spec = GameEnv.TOWER_SPECS[type_id]
        self.range = spec["range"]
        self.damage = spec["damage"]
        self.fire_rate = spec["fire_rate"]
        self.color = spec["color"]
        self.cooldown = 0

    def update(self, enemies):
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        
        target = self.find_target(enemies)
        if target:
            self.fire(target)

    def find_target(self, enemies):
        for enemy in enemies:
            dist = math.hypot(self.pos[0] - enemy.pos[0], self.pos[1] - enemy.pos[1])
            if dist <= self.range:
                return enemy
        return None

    def fire(self, target):
        self.cooldown = self.fire_rate
        projectile_color = (100, 200, 255) if self.type_id == 1 else (255, 200, 100)
        self.env.projectiles.append(Projectile(self.pos, target, self.damage, projectile_color))
        # sfx: tower_fire

    def draw(self, screen):
        base_size = 10
        top_size = 6
        draw_pos = (self.pos[0], self.pos[1] - 5)
        pygame.gfxdraw.filled_circle(screen, draw_pos[0], draw_pos[1], base_size, self.color)
        pygame.gfxdraw.aacircle(screen, draw_pos[0], draw_pos[1], base_size, tuple(c*0.8 for c in self.color))
        pygame.gfxdraw.filled_circle(screen, draw_pos[0], draw_pos[1]-2, top_size, tuple(min(255, c*1.2) for c in self.color))

class Projectile:
    def __init__(self, start_pos, target, damage, color):
        self.pos = list(start_pos)
        self.target = target
        self.damage = damage
        self.color = color
        self.speed = 8
        self.hit = False
        self.lifespan = 120 # Failsafe

    def update(self):
        self.lifespan -= 1
        if self.target.health <= 0: # Target died
            self.hit = True # Mark for removal
            return
            
        direction = (self.target.pos[0] - self.pos[0], self.target.pos[1] - self.pos[1])
        dist = math.hypot(*direction)
        
        if dist < self.speed:
            self.hit = True
        else:
            norm_dir = (direction[0] / dist, direction[1] / dist)
            self.pos[0] += norm_dir[0] * self.speed
            self.pos[1] += norm_dir[1] * self.speed
            
    def is_dead(self):
        return self.lifespan <= 0

    def draw(self, screen):
        draw_pos = (int(self.pos[0]), int(self.pos[1]))
        pygame.gfxdraw.filled_circle(screen, draw_pos[0], draw_pos[1], 3, self.color)
        pygame.gfxdraw.aacircle(screen, draw_pos[0], draw_pos[1], 3, (255, 255, 255))

class Particle:
    def __init__(self, pos, color, speed_mult):
        self.pos = list(pos)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3) * speed_mult
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = random.randint(10, 25)
        self.color = color
        self.size = random.randint(2, 4)

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.lifespan -= 1
        self.vel[0] *= 0.95
        self.vel[1] *= 0.95

    def draw(self, screen):
        alpha = max(0, int(255 * (self.lifespan / 25)))
        color_with_alpha = self.color + (alpha,)
        pygame.gfxdraw.filled_circle(screen, int(self.pos[0]), int(self.pos[1]), self.size, color_with_alpha)

if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.init()
    screen_human = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Tower Defense")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Key mapping
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    print(env.user_guide)

    while not done:
        # --- Human input ---
        movement = 0
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        for key, move_action in key_map.items():
            if keys[key]:
                movement = move_action
                break # Prioritize one movement key
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        
        # --- Environment step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # --- Render to screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen_human.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Match the environment's intended framerate
        
    print(f"Game Over! Final Score: {info['score']:.2f}, Final Wave: {info['wave']}")
    pygame.quit()