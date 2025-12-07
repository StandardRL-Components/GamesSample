
# Generated: 2025-08-27T12:35:26.720114
# Source Brief: brief_00094.md
# Brief Index: 94

        
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


# Helper classes for game entities
class Enemy:
    def __init__(self, pos, health, speed, wave_num, path_start_index=0):
        self.pos = pygame.math.Vector2(pos)
        self.max_health = health
        self.health = health
        self.speed = speed
        self.path_index = path_start_index
        self.alive = True
        self.distance_traveled = 0

    def move(self, path, dt):
        if not self.alive or self.path_index >= len(path) -1:
            return
        
        target_pos = pygame.math.Vector2(path[self.path_index + 1])
        direction = (target_pos - self.pos).normalize()
        distance_to_target = self.pos.distance_to(target_pos)
        
        move_distance = self.speed * dt
        
        if move_distance >= distance_to_target:
            self.pos = target_pos
            self.path_index += 1
            self.distance_traveled += distance_to_target
        else:
            self.pos += direction * move_distance
            self.distance_traveled += move_distance

    def draw(self, surface):
        # Body
        pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 8, (220, 40, 40))
        pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), 8, (255, 100, 100))
        
        # Health bar
        if self.health < self.max_health:
            bar_width = 16
            bar_height = 4
            health_pct = self.health / self.max_health
            fill_width = int(bar_width * health_pct)
            
            bg_rect = pygame.Rect(self.pos.x - bar_width // 2, self.pos.y - 18, bar_width, bar_height)
            health_rect = pygame.Rect(self.pos.x - bar_width // 2, self.pos.y - 18, fill_width, bar_height)
            
            pygame.draw.rect(surface, (50, 50, 50), bg_rect)
            pygame.draw.rect(surface, (100, 255, 100), health_rect)


class Tower:
    def __init__(self, grid_pos, grid_size, tower_type, stats):
        self.grid_pos = grid_pos
        self.pos = pygame.math.Vector2(
            (grid_pos[0] + 0.5) * grid_size, 
            (grid_pos[1] + 0.5) * grid_size
        )
        self.type = tower_type
        self.range = stats['range']
        self.damage = stats['damage']
        self.fire_rate = stats['fire_rate']
        self.color = stats['color']
        self.cooldown = 0

    def update(self, dt):
        if self.cooldown > 0:
            self.cooldown -= dt

    def can_fire(self):
        return self.cooldown <= 0

    def reset_cooldown(self):
        self.cooldown = 1 / self.fire_rate

    def draw(self, surface, grid_size):
        rect = pygame.Rect(self.grid_pos[0] * grid_size, self.grid_pos[1] * grid_size, grid_size, grid_size)
        
        # Glow effect
        glow_surf = pygame.Surface((grid_size*2, grid_size*2), pygame.SRCALPHA)
        pygame.gfxdraw.filled_circle(glow_surf, grid_size, grid_size, int(grid_size * 0.6), (*self.color, 50))
        surface.blit(glow_surf, (rect.centerx - grid_size, rect.centery - grid_size))

        # Base
        pygame.draw.rect(surface, self.color, rect.inflate(-8, -8), border_radius=3)
        pygame.draw.rect(surface, tuple(min(255, c+50) for c in self.color), rect.inflate(-8, -8), 2, border_radius=3)


class Projectile:
    def __init__(self, start_pos, target_enemy, damage, speed, p_type):
        self.pos = pygame.math.Vector2(start_pos)
        self.target = target_enemy
        # Aim at the center of the enemy's current position
        self.target_pos = pygame.math.Vector2(target_enemy.pos)
        self.damage = damage
        self.speed = speed
        self.type = p_type
        self.alive = True
        
        direction = (self.target_pos - self.pos)
        if direction.length() > 0:
            self.vel = direction.normalize() * speed
        else:
            self.vel = pygame.math.Vector2(0, -speed) # Failsafe

    def update(self, dt):
        self.pos += self.vel * dt
        if self.pos.distance_to(self.target_pos) < 10:
            self.alive = False

    def draw(self, surface):
        if self.type == 'basic':
            color = (255, 255, 255)
            pygame.draw.circle(surface, color, (int(self.pos.x), int(self.pos.y)), 3)
        elif self.type == 'slow':
            color = (200, 255, 255)
            pygame.gfxdraw.filled_circle(surface, int(self.pos.x), int(self.pos.y), 4, color)
            pygame.gfxdraw.aacircle(surface, int(self.pos.x), int(self.pos.y), 4, (150, 200, 255))


class Particle:
    def __init__(self, pos, color, lifetime, speed, size):
        self.pos = pygame.math.Vector2(pos)
        self.vel = pygame.math.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * speed
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime
        self.size = size

    def update(self, dt):
        self.pos += self.vel * dt
        self.lifetime -= dt
        self.vel *= 0.95 # Damping

    def draw(self, surface):
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        if alpha > 0:
            s = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.color, alpha), (self.size, self.size), self.size)
            surface.blit(s, (int(self.pos.x - self.size), int(self.pos.y - self.size)))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    user_guide = "Controls: Arrows to move cursor. Space to place Basic Tower, Shift to place Slow Tower."
    game_description = "Defend your base from waves of enemies by strategically placing towers."
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.W, self.H = 640, 400
        self.GRID_SIZE = 40
        self.MAX_WAVES = 10
        self.MAX_STEPS = 10000 # Sufficient for 10 waves

        self.observation_space = Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.FONT_S = pygame.font.SysFont("Consolas", 16, bold=True)
        self.FONT_L = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self._define_colors()
        self._define_game_parameters()
        self._define_level_geometry()
        
        self.reset()
        self.validate_implementation()

    def _define_colors(self):
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_PATH = (40, 50, 60)
        self.COLOR_GRID = (30, 40, 50)
        self.COLOR_BASE = (60, 180, 70)
        self.TOWER_STATS = {
            'basic': {'cost': 100, 'range': 100, 'damage': 10, 'fire_rate': 2, 'color': (80, 150, 255)},
            'slow': {'cost': 150, 'range': 80, 'damage': 5, 'fire_rate': 1, 'color': (255, 220, 80), 'slow_factor': 0.5}
        }

    def _define_game_parameters(self):
        self.INITIAL_BASE_HEALTH = 100
        self.INITIAL_RESOURCES = 250
        self.INTER_WAVE_DELAY = 5 * 30 # 5 seconds at 30fps
        self.ENEMY_SPAWN_DELAY = 0.5 * 30 # 0.5 seconds

    def _define_level_geometry(self):
        self.path_waypoints = [
            (-20, 100), (100, 100), (100, 300), (300, 300), 
            (300, 100), (500, 100), (500, 300), (self.W + 20, 300)
        ]
        self.base_pos = (self.W - self.GRID_SIZE, 300 - self.GRID_SIZE/2)
        
        self.placement_tiles = []
        path_rects = []
        for i in range(len(self.path_waypoints) - 1):
            p1 = self.path_waypoints[i]
            p2 = self.path_waypoints[i+1]
            path_rects.append(pygame.Rect(min(p1[0], p2[0]) - self.GRID_SIZE, min(p1[1], p2[1]) - self.GRID_SIZE,
                                          abs(p1[0]-p2[0]) + self.GRID_SIZE*2, abs(p1[1]-p2[1]) + self.GRID_SIZE*2))

        for x in range(self.W // self.GRID_SIZE):
            for y in range(self.H // self.GRID_SIZE):
                tile_rect = pygame.Rect(x * self.GRID_SIZE, y * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
                on_path = any(r.colliderect(tile_rect) for r in path_rects)
                if not on_path and y < (self.H // self.GRID_SIZE - 1): # Avoid bottom row for UI
                    self.placement_tiles.append((x, y))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)
        
        self.steps = 0
        self.game_over = False
        self.game_won = False
        self.step_reward = 0.0
        
        self.cursor_pos = [3, 5]
        self.prev_action = [0, 0, 0]
        
        self.base_health = self.INITIAL_BASE_HEALTH
        self.resources = self.INITIAL_RESOURCES
        self.wave_number = 0
        self.wave_ongoing = False
        self.wave_timer = self.INTER_WAVE_DELAY // 3 # Faster start
        self.enemies_to_spawn = 0
        self.enemy_spawn_timer = 0
        
        self.towers = []
        self.enemies = []
        self.projectiles = []
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.step_reward = -0.001 # Small time penalty
        dt = self.clock.tick(self.metadata["render_fps"]) / 1000.0 * 30 # Delta time scaled to 30fps logic
        
        self._handle_input(action)
        self._update_game_logic(dt)
        
        self.steps += 1
        terminated = self._check_termination()
        reward = self.step_reward
        self.prev_action = action
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_press = space_held and not (self.prev_action[1] == 1)
        shift_press = shift_held and not (self.prev_action[2] == 1)
        
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.H // self.GRID_SIZE - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.W // self.GRID_SIZE - 1: self.cursor_pos[0] += 1

        if space_press: self._place_tower('basic')
        if shift_press: self._place_tower('slow')

    def _place_tower(self, tower_type):
        cost = self.TOWER_STATS[tower_type]['cost']
        pos = tuple(self.cursor_pos)
        is_valid_tile = pos in self.placement_tiles
        is_unoccupied = not any(t.grid_pos == pos for t in self.towers)

        if self.resources >= cost and is_valid_tile and is_unoccupied:
            self.resources -= cost
            self.towers.append(Tower(pos, self.GRID_SIZE, tower_type, self.TOWER_STATS[tower_type]))
            # Sound: PlaceTower.wav

    def _update_game_logic(self, dt):
        self._update_waves(dt)
        self._update_towers(dt)
        self._update_enemies(dt)
        self._update_projectiles(dt)
        self._update_particles(dt)

    def _update_waves(self, dt):
        if self.game_over: return
        if not self.wave_ongoing:
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.wave_number < self.MAX_WAVES:
                self._start_next_wave()
        else:
            self.enemy_spawn_timer -= 1
            if self.enemy_spawn_timer <= 0 and self.enemies_to_spawn > 0:
                self._spawn_enemy()
                self.enemies_to_spawn -= 1
                self.enemy_spawn_timer = self.ENEMY_SPAWN_DELAY
            
            if self.enemies_to_spawn == 0 and not self.enemies:
                self.wave_ongoing = False
                self.wave_timer = self.INTER_WAVE_DELAY

    def _start_next_wave(self):
        self.wave_number += 1
        self.wave_ongoing = True
        self.enemies_to_spawn = 5 + self.wave_number * 2

    def _spawn_enemy(self):
        health = 50 * (1.1 ** self.wave_number)
        speed = 30 * (1.05 ** self.wave_number)
        self.enemies.append(Enemy(self.path_waypoints[0], health, speed, self.wave_number))

    def _update_towers(self, dt):
        for tower in self.towers:
            tower.update(dt)
            if tower.can_fire():
                # Target enemy that is furthest along the path
                target = None
                max_dist = -1
                for enemy in self.enemies:
                    if tower.pos.distance_to(enemy.pos) <= tower.range:
                        if enemy.distance_traveled > max_dist:
                            max_dist = enemy.distance_traveled
                            target = enemy
                
                if target:
                    tower.reset_cooldown()
                    self.projectiles.append(Projectile(tower.pos, target, tower.damage, 400, tower.type))
                    # Sound: Laser.wav

    def _update_enemies(self, dt):
        for enemy in self.enemies[:]:
            enemy.move(self.path_waypoints, dt)
            if enemy.path_index >= len(self.path_waypoints) - 1:
                self.base_health -= 10
                self.step_reward -= 10
                enemy.alive = False
                # Sound: BaseDamage.wav
            
            if not enemy.alive:
                self.enemies.remove(enemy)

    def _update_projectiles(self, dt):
        for p in self.projectiles[:]:
            p.update(dt)
            if p.target.alive and p.pos.distance_to(p.target.pos) < 10:
                p.alive = False
                p.target.health -= p.damage
                self.step_reward += 0.1
                # Sound: Hit.wav

                # Apply slow effect
                if p.type == 'slow':
                    p.target.speed *= self.TOWER_STATS['slow']['slow_factor']

                if p.target.health <= 0:
                    p.target.alive = False
                    self.step_reward += 1
                    self.resources += 10 + self.wave_number
                    # Sound: EnemyExplode.wav
                    for _ in range(15):
                        self.particles.append(Particle(p.target.pos, (255, 100, 100), 0.5, 100, 3))
            
            if not p.alive or p.pos.x < 0 or p.pos.x > self.W or p.pos.y < 0 or p.pos.y > self.H:
                self.projectiles.remove(p)

    def _update_particles(self, dt):
        for particle in self.particles[:]:
            particle.update(dt)
            if particle.lifetime <= 0:
                self.particles.remove(particle)

    def _check_termination(self):
        if not self.game_over:
            if self.base_health <= 0:
                self.game_over = True
                self.game_won = False
                self.step_reward += -100
            elif self.wave_number >= self.MAX_WAVES and not self.enemies and not self.wave_ongoing:
                self.game_over = True
                self.game_won = True
                self.step_reward += 100
        
        return self.game_over or self.steps >= self.MAX_STEPS

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        for tower in self.towers: tower.draw(self.screen, self.GRID_SIZE)
        for enemy in self.enemies: enemy.draw(self.screen)
        for p in self.projectiles: p.draw(self.screen)
        for particle in self.particles: particle.draw(self.screen)
        self._render_cursor()
        self._render_ui()
        if self.game_over: self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_background(self):
        for pos in self.placement_tiles:
            rect = pygame.Rect(pos[0] * self.GRID_SIZE, pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        if len(self.path_waypoints) > 1:
            pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, self.GRID_SIZE)
        
        # Base
        base_rect = pygame.Rect(self.base_pos[0], self.base_pos[1], self.GRID_SIZE, self.GRID_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_BASE, base_rect, border_radius=5)
        pygame.draw.rect(self.screen, (200, 255, 200), base_rect, 3, border_radius=5)

    def _render_cursor(self):
        pos = tuple(self.cursor_pos)
        rect = pygame.Rect(pos[0] * self.GRID_SIZE, pos[1] * self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE)
        
        is_valid = pos in self.placement_tiles and not any(t.grid_pos == pos for t in self.towers)
        color = (0, 255, 0, 100) if is_valid else (255, 0, 0, 100)
        
        cursor_surf = pygame.Surface((self.GRID_SIZE, self.GRID_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(cursor_surf, color, cursor_surf.get_rect(), border_radius=3)
        pygame.draw.rect(cursor_surf, (255,255,255,150), cursor_surf.get_rect(), 2, border_radius=3)
        self.screen.blit(cursor_surf, rect.topleft)

    def _render_ui(self):
        # UI Background
        ui_rect = pygame.Rect(0, self.H - 30, self.W, 30)
        pygame.draw.rect(self.screen, (10, 15, 20), ui_rect)
        pygame.draw.line(self.screen, (50, 60, 70), (0, self.H-30), (self.W, self.H-30), 1)

        # Health
        health_text = self.FONT_S.render(f"Base HP: {max(0, self.base_health)}/{self.INITIAL_BASE_HEALTH}", True, (100, 220, 100))
        self.screen.blit(health_text, (10, self.H - 22))

        # Resources
        res_text = self.FONT_S.render(f"Resources: ${self.resources}", True, (255, 220, 80))
        self.screen.blit(res_text, (220, self.H - 22))

        # Wave
        wave_text = self.FONT_S.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, (150, 150, 255))
        self.screen.blit(wave_text, (400, self.H - 22))
        
        # Next wave timer
        if not self.wave_ongoing and not self.game_over and self.wave_number < self.MAX_WAVES:
            timer_sec = math.ceil(self.wave_timer / self.metadata['render_fps'])
            timer_text = self.FONT_S.render(f"Next wave in: {timer_sec}", True, (200, 200, 200))
            self.screen.blit(timer_text, (self.W - 160, self.H - 22))

    def _render_game_over(self):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        msg = "VICTORY!" if self.game_won else "GAME OVER"
        color = (100, 255, 100) if self.game_won else (255, 100, 100)
        text = self.FONT_L.render(msg, True, color)
        text_rect = text.get_rect(center=(self.W // 2, self.H // 2))
        self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "resources": self.resources,
            "base_health": self.base_health,
            "wave": self.wave_number,
            "steps": self.steps,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.H, self.W, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.H, self.W, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")