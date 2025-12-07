
# Generated: 2025-08-28T04:21:08.073699
# Source Brief: brief_05222.md
# Brief Index: 5222

        
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
        "Controls: Use arrow keys to move the selector. Press Space to build a tower on the selected tile."
    )

    game_description = (
        "A classic tower defense game. Place towers to defend your base from waves of incoming enemies."
    )

    auto_advance = True

    # --- Constants ---
    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_PATH = (60, 70, 100)
    COLOR_BASE = (0, 150, 100)
    COLOR_BASE_DMG = (255, 100, 100)
    COLOR_SELECTOR = (255, 255, 0)
    COLOR_TOWER = (255, 200, 0)
    COLOR_TOWER_GUN = (200, 150, 0)
    COLOR_ENEMY = (200, 50, 50)
    COLOR_PROJECTILE = (100, 200, 255)
    COLOR_TEXT = (220, 220, 240)
    COLOR_HEALTH_GREEN = (0, 200, 0)
    COLOR_HEALTH_RED = (200, 0, 0)
    
    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 8
    CELL_SIZE = 40
    GRID_WIDTH = GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    GRID_TOP_LEFT = ((SCREEN_WIDTH - GRID_WIDTH) // 2, (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20)

    # Game Mechanics
    MAX_STEPS = 15000  # ~8 minutes at 30fps
    FPS = 30
    BASE_MAX_HEALTH = 100
    STARTING_RESOURCES = 150
    TOWER_COST = 50
    TOWER_RANGE = 90
    TOWER_DAMAGE = 8
    TOWER_FIRE_RATE = 1.0 # shots per second
    ENEMY_KILL_REWARD = 10
    WAVE_COUNT = 10
    WAVE_COMPLETION_BONUS = 100
    WIN_BONUS = 500
    WAVE_INTERVAL = 300 # frames (10 seconds)

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
        self.font_small = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_huge = pygame.font.SysFont("Consolas", 48, bold=True)
        
        self._define_path()
        
        self.selector_pos = [0, 0]
        self.previous_space_held = False
        self.selector_move_cooldown = 0
        
        self.reset()
        
        self.validate_implementation()

    def _define_path(self):
        # Path in grid coordinates
        self.path_grid_coords = [
            (-1, 2), (0, 2), (1, 2), (2, 2), (2, 3), (2, 4), (2, 5), (3, 5),
            (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (6, 1), (7, 1), (8, 1)
        ]
        self.path_world_coords = [self._grid_to_world(x, y) for x, y in self.path_grid_coords]
        self.path_tiles = set(self.path_grid_coords[1:-1])
        self.base_pos = self.path_world_coords[-1]
        self.spawn_pos = self.path_world_coords[0]

    def _grid_to_world(self, x, y):
        return (
            self.GRID_TOP_LEFT[0] + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_TOP_LEFT[1] + y * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = self.BASE_MAX_HEALTH
        self.base_damage_flash = 0
        self.resources = self.STARTING_RESOURCES
        
        self.current_wave = 0
        self.wave_timer = self.WAVE_INTERVAL // 2
        self.wave_active = False
        self.enemies_to_spawn = []
        
        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []
        
        self.selector_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.previous_space_held = False
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.001 # Small penalty for existing
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Player Actions ---
        self._handle_actions(movement, space_held)
        
        # --- Update Game State ---
        self._update_wave_system()
        reward += self._update_towers()
        reward += self._update_projectiles()
        reward += self._update_enemies()
        self._update_particles()

        self.steps += 1
        
        # --- Check Termination ---
        if not self.win and self.current_wave > self.WAVE_COUNT and not self.enemies and not self.enemies_to_spawn:
            self.win = True
            reward += self.WIN_BONUS
            self.game_over = True
        
        if self.base_health <= 0:
            self.game_over = True
            reward -= 100 # Large penalty for losing
            
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_actions(self, movement, space_held):
        # Selector movement
        if self.selector_move_cooldown > 0:
            self.selector_move_cooldown -= 1
        
        if movement != 0 and self.selector_move_cooldown == 0:
            dx, dy = 0, 0
            if movement == 1: dy = -1
            elif movement == 2: dy = 1
            elif movement == 3: dx = -1
            elif movement == 4: dx = 1
            
            self.selector_pos[0] = np.clip(self.selector_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.selector_pos[1] = np.clip(self.selector_pos[1] + dy, 0, self.GRID_SIZE - 1)
            self.selector_move_cooldown = 5 # frames

        # Place tower
        if space_held and not self.previous_space_held:
            self._place_tower()
        self.previous_space_held = space_held

    def _place_tower(self):
        grid_pos = tuple(self.selector_pos)
        if self.resources < self.TOWER_COST: return # sfx: error
        if grid_pos in self.path_tiles: return # sfx: error
        if any(t.grid_pos == grid_pos for t in self.towers): return # sfx: error

        self.resources -= self.TOWER_COST
        world_pos = self._grid_to_world(*grid_pos)
        self.towers.append(Tower(world_pos, grid_pos))
        # sfx: build_tower

    def _update_wave_system(self):
        if self.wave_active and not self.enemies and not self.enemies_to_spawn:
            self.wave_active = False
            self.wave_timer = self.WAVE_INTERVAL
            if self.current_wave > 0:
                self.score += self.WAVE_COMPLETION_BONUS
        
        if not self.wave_active:
            self.wave_timer -= 1
            if self.wave_timer <= 0 and self.current_wave < self.WAVE_COUNT:
                self._start_next_wave()

        if self.enemies_to_spawn and self.steps % 15 == 0: # Spawn one enemy every 0.5s
            self.enemies.append(self.enemies_to_spawn.pop(0))

    def _start_next_wave(self):
        self.current_wave += 1
        self.wave_active = True
        num_enemies = 3 + (self.current_wave - 1) * 2
        health = 10 * (1.15 ** (self.current_wave - 1))
        speed = 0.8 * (1.05 ** (self.current_wave - 1))
        
        for _ in range(num_enemies):
            self.enemies_to_spawn.append(Enemy(self.spawn_pos, list(self.path_world_coords), health, speed))
        # sfx: wave_start

    def _update_towers(self):
        reward = 0
        for tower in self.towers:
            tower.update()
            if tower.can_fire():
                target = tower.find_target(self.enemies)
                if target:
                    tower.fire(target)
                    self.projectiles.append(Projectile(tower.pos, target, self.TOWER_DAMAGE))
                    # sfx: tower_shoot
        return reward

    def _update_projectiles(self):
        reward = 0
        for p in self.projectiles[:]:
            p.update()
            if p.hit:
                self.projectiles.remove(p)
                reward += 0.1 # Reward for hitting
                p.target.take_damage(p.damage)
                # sfx: enemy_hit
                self._create_particles(p.pos, 5, self.COLOR_PROJECTILE, 1, 3)
                if p.target.health <= 0:
                    reward += 1 # Reward for destroying
                    self.resources += self.ENEMY_KILL_REWARD
                    self.enemies.remove(p.target)
                    # sfx: enemy_explode
                    self._create_particles(p.target.pos, 20, self.COLOR_ENEMY, 2, 4)
        return reward

    def _update_enemies(self):
        reward = 0
        for enemy in self.enemies[:]:
            enemy.update()
            if enemy.reached_base:
                self.enemies.remove(enemy)
                self.base_health -= 10
                self.base_damage_flash = 15 # frames
                reward -= 10 # Penalty for base damage
                # sfx: base_damage
                self._create_particles(self.base_pos, 30, self.COLOR_BASE_DMG, 3, 5)
        self.base_health = max(0, self.base_health)
        return reward

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, min_speed, max_speed):
        for _ in range(count):
            self.particles.append(Particle(pos, color, min_speed, max_speed))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Game Elements ---
        self._render_grid_and_path()
        self._render_base()
        
        for enemy in self.enemies: enemy.draw(self.screen)
        for tower in self.towers: tower.draw(self.screen)
        for proj in self.projectiles: proj.draw(self.screen)
        for part in self.particles: part.draw(self.screen)
        
        self._render_selector()
        
        # --- Render UI ---
        self._render_ui()
        
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_path(self):
        gx, gy = self.GRID_TOP_LEFT
        # Draw path tiles
        for x_idx, y_idx in self.path_tiles:
            rect = (gx + x_idx * self.CELL_SIZE, gy + y_idx * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_PATH, rect)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_v = (gx + i * self.CELL_SIZE, gy)
            end_v = (gx + i * self.CELL_SIZE, gy + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_v, end_v)
            # Horizontal
            start_h = (gx, gy + i * self.CELL_SIZE)
            end_h = (gx + self.GRID_WIDTH, gy + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_h, end_h)

    def _render_base(self):
        if self.base_damage_flash > 0:
            self.base_damage_flash -= 1
            color = self.COLOR_BASE_DMG
        else:
            color = self.COLOR_BASE
        
        base_rect = pygame.Rect(0, 0, self.CELL_SIZE, self.CELL_SIZE)
        base_rect.center = self.base_pos
        pygame.draw.rect(self.screen, color, base_rect, border_radius=5)
        
        # Health bar
        bar_w = self.CELL_SIZE
        bar_h = 5
        bar_x = base_rect.left
        bar_y = base_rect.top - 10
        health_w = (self.base_health / self.BASE_MAX_HEALTH) * bar_w
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (bar_x, bar_y, health_w, bar_h))

    def _render_selector(self):
        sel_x, sel_y = self.selector_pos
        world_x = self.GRID_TOP_LEFT[0] + sel_x * self.CELL_SIZE
        world_y = self.GRID_TOP_LEFT[1] + sel_y * self.CELL_SIZE
        rect = (world_x, world_y, self.CELL_SIZE, self.CELL_SIZE)
        
        # Pulsing alpha
        alpha = 100 + int(math.sin(self.steps * 0.2) * 50)
        
        # Check validity for color coding
        grid_pos = tuple(self.selector_pos)
        can_build = self.resources >= self.TOWER_COST and grid_pos not in self.path_tiles and not any(t.grid_pos == grid_pos for t in self.towers)
        color = (0, 255, 0) if can_build else self.COLOR_SELECTOR

        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, (*color, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.rect(s, (*color, 200), (0, 0, self.CELL_SIZE, self.CELL_SIZE), 2)
        self.screen.blit(s, (world_x, world_y))

    def _render_ui(self):
        # Wave
        wave_text = f"WAVE: {self.current_wave}/{self.WAVE_COUNT}"
        text_surf = self.font_large.render(wave_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 10))

        # Base Health
        health_text = f"BASE: {self.base_health}%"
        text_surf = self.font_large.render(health_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (self.SCREEN_WIDTH - text_surf.get_width() - 10, 10))

        # Resources
        res_text = f"RESOURCES: ${self.resources}"
        text_surf = self.font_large.render(res_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, ((self.SCREEN_WIDTH - text_surf.get_width()) // 2, self.SCREEN_HEIGHT - 30))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        text_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(text_surf, (10, 40))

        # Wave Timer
        if not self.wave_active and self.current_wave < self.WAVE_COUNT:
            timer_sec = self.wave_timer / self.FPS
            timer_text = f"NEXT WAVE IN {timer_sec:.1f}s"
            text_surf = self.font_large.render(timer_text, True, self.COLOR_SELECTOR)
            self.screen.blit(text_surf, ((self.SCREEN_WIDTH - text_surf.get_width()) // 2, 10))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "VICTORY!" if self.win else "GAME OVER"
        text_surf = self.font_huge.render(message, True, self.COLOR_SELECTOR if self.win else self.COLOR_ENEMY)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "base_health": self.base_health,
            "resources": self.resources,
        }

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
    def __init__(self, pos, path, health, speed):
        self.pos = list(pos)
        self.path = path
        self.path_index = 1
        self.max_health = health
        self.health = health
        self.speed = speed
        self.radius = 8
        self.reached_base = False

    def update(self):
        if self.path_index >= len(self.path):
            self.reached_base = True
            return

        target_pos = self.path[self.path_index]
        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.pos = list(target_pos)
            self.path_index += 1
        else:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed
    
    def take_damage(self, amount):
        self.health -= amount
        self.health = max(0, self.health)

    def draw(self, screen):
        # Body
        pygame.gfxdraw.filled_circle(screen, int(self.pos[0]), int(self.pos[1]), self.radius, GameEnv.COLOR_ENEMY)
        pygame.gfxdraw.aacircle(screen, int(self.pos[0]), int(self.pos[1]), self.radius, GameEnv.COLOR_ENEMY)
        
        # Health bar
        bar_w = self.radius * 2
        bar_h = 3
        bar_x = self.pos[0] - self.radius
        bar_y = self.pos[1] - self.radius - 8
        health_w = (self.health / self.max_health) * bar_w
        pygame.draw.rect(screen, GameEnv.COLOR_HEALTH_RED, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(screen, GameEnv.COLOR_HEALTH_GREEN, (bar_x, bar_y, health_w, bar_h))

class Tower:
    def __init__(self, pos, grid_pos):
        self.pos = pos
        self.grid_pos = grid_pos
        self.cooldown = 0
        self.fire_rate_frames = GameEnv.FPS / GameEnv.TOWER_FIRE_RATE
        self.angle = 0

    def update(self):
        if self.cooldown > 0:
            self.cooldown -= 1

    def can_fire(self):
        return self.cooldown <= 0

    def find_target(self, enemies):
        valid_targets = [e for e in enemies if math.hypot(e.pos[0] - self.pos[0], e.pos[1] - self.pos[1]) < GameEnv.TOWER_RANGE]
        if not valid_targets:
            return None
        # Target enemy closest to the base (highest path_index)
        return max(valid_targets, key=lambda e: e.path_index)

    def fire(self, target):
        self.cooldown = self.fire_rate_frames
        dx = target.pos[0] - self.pos[0]
        dy = target.pos[1] - self.pos[1]
        self.angle = math.degrees(math.atan2(-dy, dx))

    def draw(self, screen):
        # Base
        pygame.draw.rect(screen, GameEnv.COLOR_TOWER, (self.pos[0]-12, self.pos[1]-12, 24, 24), border_radius=4)
        
        # Gun
        gun_length = 15
        end_x = self.pos[0] + gun_length * math.cos(math.radians(self.angle))
        end_y = self.pos[1] - gun_length * math.sin(math.radians(self.angle))
        pygame.draw.line(screen, GameEnv.COLOR_TOWER_GUN, self.pos, (end_x, end_y), 6)

class Projectile:
    def __init__(self, pos, target, damage):
        self.pos = list(pos)
        self.target = target
        self.damage = damage
        self.speed = 8
        self.hit = False

    def update(self):
        dx = self.target.pos[0] - self.pos[0]
        dy = self.target.pos[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.hit = True
        else:
            self.pos[0] += (dx / dist) * self.speed
            self.pos[1] += (dy / dist) * self.speed
    
    def draw(self, screen):
        pygame.draw.rect(screen, GameEnv.COLOR_PROJECTILE, (int(self.pos[0])-2, int(self.pos[1])-2, 4, 4))


class Particle:
    def __init__(self, pos, color, min_speed, max_speed):
        self.pos = list(pos)
        self.color = color
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(min_speed, max_speed)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]
        self.lifespan = random.randint(10, 20)
        self.max_lifespan = self.lifespan

    def update(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.95 # friction
        self.vel[1] *= 0.95
        self.lifespan -= 1

    def is_dead(self):
        return self.lifespan <= 0

    def draw(self, screen):
        alpha = int(255 * (self.lifespan / self.max_lifespan))
        color = (*self.color, alpha)
        s = pygame.Surface((3, 3), pygame.SRCALPHA)
        s.fill(color)
        screen.blit(s, (int(self.pos[0])-1, int(self.pos[1])-1))