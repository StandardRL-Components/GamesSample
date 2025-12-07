
# Generated: 2025-08-28T05:43:58.181078
# Source Brief: brief_05674.md
# Brief Index: 5674

        
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
        "Controls: Arrows to move cursor. Shift to cycle building type. Space to place selected building."
    )

    game_description = (
        "Defend your base from alien waves in this isometric RTS. Place towers and walls strategically to survive."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame Setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # Constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 25, 20
        self.TILE_W, self.TILE_H = 32, 16
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Colors
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 35, 60)
        self.COLOR_BASE = (0, 150, 255)
        self.COLOR_TOWER = (0, 200, 100)
        self.COLOR_WALL = (100, 100, 120)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PROJECTILE = (100, 200, 255)
        self.COLOR_CURSOR_VALID = (255, 255, 0)
        self.COLOR_CURSOR_INVALID = (255, 0, 0)
        self.COLOR_TEXT = (220, 220, 220)
        
        # Game Settings
        self.BASE_MAX_HEALTH = 100
        self.INITIAL_RESOURCES = 75
        self.MAX_STEPS = 3000 # Approx 100 seconds at 30fps
        self.MAX_WAVES = 10
        self.WAVE_DELAY_FRAMES = 150 # 5 seconds

        # Building Costs
        self.BUILDING_COSTS = {"TOWER": 25, "WALL": 10}
        self.BUILDING_TYPES = ["TOWER", "WALL"]

        # Action handling state
        self.prev_space_held = False
        self.prev_shift_held = False

        # Initialize state variables
        self.reset()
        
        # This will run once to ensure the implementation is correct
        # self.validate_implementation()

    def _iso_to_screen(self, gx, gy):
        sx = self.ORIGIN_X + (gx - gy) * self.TILE_W / 2
        sy = self.ORIGIN_Y + (gx + gy) * self.TILE_H / 2
        return int(sx), int(sy)

    def _get_tile_poly(self, gx, gy):
        x, y = self._iso_to_screen(gx, gy)
        half_w, half_h = self.TILE_W / 2, self.TILE_H / 2
        return [
            (x, y - half_h), (x + half_w, y),
            (x, y + half_h), (x - half_w, y)
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.base_core_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.base_health = self.BASE_MAX_HEALTH
        self.resources = self.INITIAL_RESOURCES
        
        self.grid = [["EMPTY" for _ in range(self.GRID_W)] for _ in range(self.GRID_H)]
        self.grid[self.base_core_pos[1]][self.base_core_pos[0]] = "BASE"

        self.cursor_pos = [self.base_core_pos[0], self.base_core_pos[1] - 3]
        self.selected_building_idx = 0

        self.aliens = []
        self.buildings = []
        self.projectiles = []
        self.particles = []

        self.wave = 0
        self.wave_timer = self.WAVE_DELAY_FRAMES // 2
        self.aliens_to_spawn = 0
        self.aliens_in_wave = 0
        
        self.path_map = None
        self._recalculate_path_map()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        if not self.game_over:
            reward += self._handle_actions(action)
            reward += self._update_game_state()
            self.steps += 1
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            if self.win:
                reward += 100
            else:
                reward += -100
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_actions(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # --- Cycle Building Type (on press) ---
        if shift_held and not self.prev_shift_held:
            self.selected_building_idx = (self.selected_building_idx + 1) % len(self.BUILDING_TYPES)
        
        # --- Place Building (on press) ---
        if space_held and not self.prev_space_held:
            build_type = self.BUILDING_TYPES[self.selected_building_idx]
            cost = self.BUILDING_COSTS[build_type]
            cx, cy = self.cursor_pos
            if self.grid[cy][cx] == "EMPTY" and self.resources >= cost:
                self.resources -= cost
                self.grid[cy][cx] = build_type
                
                if build_type == "TOWER":
                    self.buildings.append(Tower(cx, cy, self))
                elif build_type == "WALL":
                    self.buildings.append(Wall(cx, cy, self))
                
                if build_type == "WALL":
                    self._recalculate_path_map()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        return 0

    def _update_game_state(self):
        step_reward = 0

        # Wave Management
        if not self.aliens and self.aliens_to_spawn == 0:
            if self.wave > 0 and self.wave < self.MAX_WAVES:
                step_reward += 5 # Wave clear reward
            if self.wave < self.MAX_WAVES:
                self.wave_timer -= 1
                if self.wave_timer <= 0:
                    self._start_next_wave()
        
        if self.aliens_to_spawn > 0 and self.steps % 15 == 0: # Spawn one alien every 0.5s
            self._spawn_alien()
            self.aliens_to_spawn -= 1

        # Update Game Objects
        for b in self.buildings: step_reward += b.update(self.aliens)
        for p in self.projectiles[:]: p.update()
        for a in self.aliens[:]: step_reward += a.update()
        for p in self.particles[:]: p.update()

        return step_reward

    def _start_next_wave(self):
        self.wave += 1
        self.wave_timer = self.WAVE_DELAY_FRAMES
        self.aliens_in_wave = 2 + self.wave
        self.aliens_to_spawn = self.aliens_in_wave
        
    def _spawn_alien(self):
        side = self.np_random.integers(4)
        if side == 0: # Top-left
            gx, gy = self.np_random.integers(self.GRID_W // 2), 0
        elif side == 1: # Top-right
            gx, gy = self.GRID_W - 1, self.np_random.integers(self.GRID_H // 2)
        elif side == 2: # Bottom-left
            gx, gy = 0, self.np_random.integers(self.GRID_H // 2, self.GRID_H)
        else: # Bottom-right
            gx, gy = self.np_random.integers(self.GRID_W // 2, self.GRID_W), self.GRID_H - 1
        
        speed = 0.5 + self.wave * 0.05
        health = 10 + self.wave * 2
        self.aliens.append(Alien(gx, gy, health, speed, self))

    def _recalculate_path_map(self):
        self.path_map = np.full((self.GRID_H, self.GRID_W), -1, dtype=int)
        q = deque([self.base_core_pos])
        self.path_map[self.base_core_pos[1], self.base_core_pos[0]] = 0
        
        while q:
            gx, gy = q.popleft()
            dist = self.path_map[gy, gx]
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.GRID_W and 0 <= ny < self.GRID_H:
                    if self.grid[ny][nx] not in ["WALL", "TOWER", "BASE"] and self.path_map[ny, nx] == -1:
                        self.path_map[ny, nx] = dist + 1
                        q.append((nx, ny))

    def _check_termination(self):
        if self.base_health <= 0:
            return True
        if self.wave >= self.MAX_WAVES and not self.aliens and self.aliens_to_spawn == 0:
            self.win = True
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        
        # Render objects sorted by isometric Y-coordinate for correct layering
        render_queue = sorted(self.buildings + self.aliens, key=lambda obj: obj.gx + obj.gy)
        for obj in render_queue:
            obj.draw(self.screen)
        
        for p in self.projectiles: p.draw(self.screen)
        self._render_cursor()
        for p in self.particles: p.draw(self.screen)
        self._render_ui()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for gy in range(self.GRID_H):
            for gx in range(self.GRID_W):
                poly = self._get_tile_poly(gx, gy)
                pygame.draw.polygon(self.screen, self.COLOR_GRID, poly, 1)

    def _render_cursor(self):
        cx, cy = self.cursor_pos
        poly = self._get_tile_poly(cx, cy)
        
        build_type = self.BUILDING_TYPES[self.selected_building_idx]
        cost = self.BUILDING_COSTS[build_type]
        
        is_valid = self.grid[cy][cx] == "EMPTY" and self.resources >= cost
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        pygame.draw.polygon(self.screen, color, poly, 2)

    def _render_ui(self):
        # Top-left info
        wave_text = self.font_small.render(f"Wave: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        resource_text = self.font_small.render(f"Resources: {self.resources}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (10, 10))
        self.screen.blit(resource_text, (10, 30))

        # Top-right info
        build_type = self.BUILDING_TYPES[self.selected_building_idx]
        cost = self.BUILDING_COSTS[build_type]
        selected_text = self.font_small.render(f"Selected: {build_type} (Cost: {cost})", True, self.COLOR_TEXT)
        self.screen.blit(selected_text, (self.WIDTH - selected_text.get_width() - 10, 10))

        # Base health bar
        bx, by = self._iso_to_screen(self.base_core_pos[0], self.base_core_pos[1])
        bar_w, bar_h = 50, 8
        health_pct = self.base_health / self.BASE_MAX_HEALTH
        
        health_color = (0, 255, 0)
        if health_pct < 0.6: health_color = (255, 255, 0)
        if health_pct < 0.3: health_color = (255, 0, 0)

        pygame.draw.rect(self.screen, (50, 50, 50), (bx - bar_w/2, by - self.TILE_H - bar_h - 5, bar_w, bar_h))
        pygame.draw.rect(self.screen, health_color, (bx - bar_w/2, by - self.TILE_H - bar_h - 5, bar_w * health_pct, bar_h))

        # Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            text = "VICTORY" if self.win else "GAME OVER"
            rendered_text = self.font_large.render(text, True, self.COLOR_CURSOR_VALID if self.win else self.COLOR_ALIEN)
            text_rect = rendered_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(rendered_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "base_health": self.base_health,
            "resources": self.resources,
            "aliens": len(self.aliens)
        }

    def close(self):
        pygame.quit()

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

# Helper classes for game objects
class Building:
    def __init__(self, gx, gy, env):
        self.gx, self.gy = gx, gy
        self.env = env
        self.x, self.y = self.env._iso_to_screen(gx, gy)
    
    def update(self, aliens): return 0
    def draw(self, screen): pass

class Tower(Building):
    def __init__(self, gx, gy, env):
        super().__init__(gx, gy, env)
        self.range = 4 * env.TILE_W
        self.cooldown = 0
        self.max_cooldown = 30 # Fire rate
        self.turret_angle = 0

    def update(self, aliens):
        if self.cooldown > 0:
            self.cooldown -= 1
        
        target = None
        min_dist = self.range
        for alien in aliens:
            dist = math.hypot(alien.x - self.x, alien.y - self.y)
            if dist < min_dist:
                min_dist = dist
                target = alien
        
        if target:
            self.turret_angle = math.atan2(target.y - self.y, target.x - self.x)
            if self.cooldown == 0:
                # sfx: tower_shoot.wav
                self.env.projectiles.append(Projectile(self.x, self.y, target, self.env))
                self.cooldown = self.max_cooldown
        return 0

    def draw(self, screen):
        # Base
        poly = self.env._get_tile_poly(self.gx, self.gy)
        pygame.gfxdraw.filled_polygon(screen, poly, self.env.COLOR_TOWER)
        pygame.gfxdraw.aapolygon(screen, poly, self.env.COLOR_TOWER)
        
        # Turret
        turret_len = 10
        end_x = self.x + turret_len * math.cos(self.turret_angle)
        end_y = self.y + turret_len * math.sin(self.turret_angle)
        pygame.draw.line(screen, (200, 255, 200), (self.x, self.y), (end_x, end_y), 3)
        pygame.draw.circle(screen, (150, 250, 150), (self.x, self.y), 5)


class Wall(Building):
    def draw(self, screen):
        poly = self.env._get_tile_poly(self.gx, self.gy)
        darker_color = tuple(c*0.7 for c in self.env.COLOR_WALL)
        pygame.gfxdraw.filled_polygon(screen, poly, darker_color)
        pygame.gfxdraw.aapolygon(screen, poly, self.env.COLOR_WALL)

class Alien:
    def __init__(self, gx, gy, health, speed, env):
        self.env = env
        self.gx, self.gy = gx, gy
        self.x, self.y = self.env._iso_to_screen(gx, gy)
        self.target_x, self.target_y = self.x, self.y
        self.max_health = health
        self.health = health
        self.speed = speed
        self.hit_timer = 0
    
    def update(self):
        reward = 0
        if self.hit_timer > 0:
            self.hit_timer -= 1

        # Check if reached current grid target
        if math.hypot(self.x - self.target_x, self.y - self.target_y) < 2:
            min_dist = 1e9
            next_gx, next_gy = self.gx, self.gy
            
            # Find next grid cell from path map
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = self.gx + dx, self.gy + dy
                if 0 <= nx < self.env.GRID_W and 0 <= ny < self.env.GRID_H:
                    path_val = self.env.path_map[ny, nx]
                    if path_val != -1 and path_val < min_dist:
                        min_dist = path_val
                        next_gx, next_gy = nx, ny
            
            self.gx, self.gy = next_gx, next_gy
            self.target_x, self.target_y = self.env._iso_to_screen(self.gx, self.gy)

            # Check for reaching base
            if (self.gx, self.gy) == self.env.base_core_pos:
                # sfx: base_damage.wav
                self.env.base_health -= 10
                self.env.aliens.remove(self)
                reward -= 0.01 * 10
                self.env.score -= 1
                return reward
        
        # Move towards target
        angle = math.atan2(self.target_y - self.y, self.target_x - self.x)
        self.x += self.speed * math.cos(angle)
        self.y += self.speed * math.sin(angle)
        return reward

    def take_damage(self, amount):
        # sfx: alien_hit.wav
        self.health -= amount
        self.hit_timer = 5 # Flash for 5 frames
        if self.health <= 0:
            # sfx: alien_destroy.wav
            if self in self.env.aliens:
                self.env.aliens.remove(self)
            self.env.resources += 3
            self.env.score += 1
            for _ in range(15):
                self.env.particles.append(Particle(self.x, self.y, self.env.COLOR_ALIEN, self.env))
            return 1.0 # Kill reward
        return 0.1 # Hit reward

    def draw(self, screen):
        color = (255, 255, 255) if self.hit_timer > 0 else self.env.COLOR_ALIEN
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 6)
        pygame.draw.circle(screen, (0,0,0), (int(self.x), int(self.y)), 6, 1)


class Projectile:
    def __init__(self, x, y, target_alien, env):
        self.env = env
        self.x, self.y = x, y
        self.target = target_alien
        angle = math.atan2(target_alien.y - y, target_alien.x - x)
        self.speed = 8
        self.vx = self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)
        self.life = 60 # Max travel time

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        
        if self.life <= 0:
            if self in self.env.projectiles: self.env.projectiles.remove(self)
            return

        # Simple collision check
        if math.hypot(self.x - self.target.x, self.y - self.target.y) < 8:
            reward = self.target.take_damage(5)
            self.env.score += reward
            if self in self.env.projectiles: self.env.projectiles.remove(self)

    def draw(self, screen):
        pygame.draw.circle(screen, self.env.COLOR_PROJECTILE, (int(self.x), int(self.y)), 3)

class Particle:
    def __init__(self, x, y, color, env):
        self.env = env
        self.x, self.y = x, y
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 3)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        self.life = 20
        self.size = random.randint(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95
        self.life -= 1
        if self.life <= 0:
            if self in self.env.particles: self.env.particles.remove(self)

    def draw(self, screen):
        alpha = int(255 * (self.life / 20))
        color = (*self.color, alpha)
        temp_surf = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, color, (self.size, self.size), self.size)
        screen.blit(temp_surf, (int(self.x - self.size), int(self.y - self.size)))