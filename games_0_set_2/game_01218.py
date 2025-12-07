
# Generated: 2025-08-27T16:24:43.566667
# Source Brief: brief_01218.md
# Brief Index: 1218

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrows to move cursor. Space to place a Basic Tower. Shift to place an Advanced Tower."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Defend your base from waves of geometric enemies by strategically placing towers on the grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 10
    GRID_BOX_SIZE = 40
    MAX_STEPS = 18000  # 10 minutes at 30fps
    MAX_WAVES = 10

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID = (40, 44, 52)
    COLOR_PATH = (33, 37, 45)
    COLOR_BASE = (0, 120, 255)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEALTH = (40, 200, 120)
    COLOR_UI_HEALTH_BG = (80, 80, 80)
    COLOR_MONEY = (255, 200, 0)
    
    # Player/Cursor
    COLOR_CURSOR_VALID = (0, 255, 0, 100)
    COLOR_CURSOR_INVALID = (255, 0, 0, 100)

    # Towers
    TOWER_BASIC_COST = 100
    TOWER_BASIC_RANGE = 80
    TOWER_BASIC_COOLDOWN = 30 # steps
    TOWER_BASIC_DMG = 10
    TOWER_ADV_COST = 250
    TOWER_ADV_RANGE = 120
    TOWER_ADV_COOLDOWN = 50 # steps
    TOWER_ADV_DMG = 25

    # Enemies
    ENEMY_BASE_HEALTH = 50
    ENEMY_BASE_SPEED = 0.75
    ENEMY_BASE_BOUNTY = 20
    ENEMY_BASE_DAMAGE = 10

    # Rewards
    REWARD_HIT = 0.01 # Small reward for hitting
    REWARD_KILL = 1.0
    PENALTY_BASE_DMG = -5.0
    REWARD_WAVE_CLEAR = 10.0
    REWARD_WIN = 100.0

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.path_waypoints = [
            (-20, 180), (100, 180), (100, 300), (300, 300), (300, 100),
            (500, 100), (500, 260), (self.SCREEN_WIDTH + 20, 260)
        ]
        
        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        
        self.base_health = 100
        self.money = 250
        
        self.wave_number = 0
        self.wave_state = "INTERMISSION" # or "ACTIVE"
        self.intermission_timer = 150 # 5 seconds
        self.enemies_to_spawn = []

        self.enemies = []
        self.towers = []
        self.projectiles = []
        self.particles = []

        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        terminated = False

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Handle Player Action
        self._handle_action(action)

        # 2. Update Game State
        self._update_wave_manager()
        
        tower_hit_reward = self._update_towers()
        reward += tower_hit_reward

        kill_reward, base_penalty = self._update_enemies_and_projectiles()
        reward += kill_reward + base_penalty

        self._update_particles()
        
        # 3. Check for wave clear
        if self.wave_state == "ACTIVE" and not self.enemies and not self.enemies_to_spawn:
            self.wave_state = "INTERMISSION"
            self.intermission_timer = 300 # 10 seconds
            reward += self.REWARD_WAVE_CLEAR
            self.money += 100 + self.wave_number * 25
            if self.wave_number >= self.MAX_WAVES:
                self.win = True

        # 4. Update Score & Steps
        self.score += reward
        self.steps += 1
        
        # 5. Check Termination Conditions
        if self.base_health <= 0:
            self.game_over = True
            terminated = True
        elif self.win:
            reward += self.REWARD_WIN
            self.score += self.REWARD_WIN
            self.game_over = True
            terminated = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_action(self, action):
        movement, place_basic, place_advanced = action[0], action[1] == 1, action[2] == 1

        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        if movement == 2: self.cursor_pos[1] += 1 # Down
        if movement == 3: self.cursor_pos[0] -= 1 # Left
        if movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Place tower
        if place_basic: self._place_tower('basic')
        if place_advanced: self._place_tower('advanced')

    def _place_tower(self, tower_type):
        x, y = self.cursor_pos
        if self.grid[y][x] is not None: return False # Cell occupied

        cost = self.TOWER_BASIC_COST if tower_type == 'basic' else self.TOWER_ADV_COST
        if self.money >= cost:
            self.money -= cost
            tower_class = TowerBasic if tower_type == 'basic' else TowerAdvanced
            new_tower = tower_class(x * self.GRID_BOX_SIZE + self.GRID_BOX_SIZE // 2,
                                    y * self.GRID_BOX_SIZE + self.GRID_BOX_SIZE // 2)
            self.towers.append(new_tower)
            self.grid[y][x] = new_tower
            # sfx: tower_place
            self._create_particles(new_tower.pos, 15, (200, 255, 200), 20)
            return True
        # sfx: action_fail
        return False

    def _update_wave_manager(self):
        if self.wave_state == "INTERMISSION":
            self.intermission_timer -= 1
            if self.intermission_timer <= 0 and self.wave_number < self.MAX_WAVES:
                self.wave_number += 1
                self.wave_state = "ACTIVE"
                self._spawn_wave()
        elif self.wave_state == "ACTIVE":
            if self.enemies_to_spawn and self.steps % 30 == 0:
                enemy_data = self.enemies_to_spawn.pop(0)
                self.enemies.append(Enemy(self.path_waypoints, **enemy_data))

    def _spawn_wave(self):
        num_enemies = 3 + self.wave_number * 2
        health = self.ENEMY_BASE_HEALTH * (1.1 ** (self.wave_number - 1))
        speed = self.ENEMY_BASE_SPEED * (1.05 ** (self.wave_number - 1))
        bounty = int(self.ENEMY_BASE_BOUNTY * (1.1 ** (self.wave_number - 1)))
        
        for _ in range(num_enemies):
            self.enemies_to_spawn.append({
                'health': health, 'speed': speed, 'bounty': bounty
            })
        # sfx: wave_start

    def _update_towers(self):
        hit_reward = 0
        for tower in self.towers:
            tower.cooldown -= 1
            if tower.cooldown <= 0:
                target = tower.find_target(self.enemies)
                if target:
                    # sfx: tower_shoot
                    self.projectiles.append(Projectile(tower.pos, target, tower.damage, tower.proj_color))
                    tower.cooldown = tower.max_cooldown
                    hit_reward += self.REWARD_HIT
        return hit_reward

    def _update_enemies_and_projectiles(self):
        kill_reward = 0
        base_penalty = 0

        for p in self.projectiles[:]:
            p.move()
            if p.check_collision():
                # sfx: enemy_hit
                p.target.health -= p.damage
                self._create_particles(p.pos, 5, p.color, 10)
                self.projectiles.remove(p)
            elif p.is_out_of_bounds():
                self.projectiles.remove(p)

        for e in self.enemies[:]:
            if e.health <= 0:
                # sfx: enemy_explode
                kill_reward += self.REWARD_KILL
                self.money += e.bounty
                self._create_particles(e.pos, 30, (255, 100, 50), 30, 2, 5)
                self.enemies.remove(e)
                continue

            e.move()
            if e.reached_end:
                # sfx: base_damage
                self.base_health -= self.ENEMY_BASE_DAMAGE
                base_penalty += self.PENALTY_BASE_DMG
                self._create_particles((self.SCREEN_WIDTH - 10, self.path_waypoints[-1][1]), 40, (255, 50, 50), 40, 3, 7)
                self.enemies.remove(e)

        self.base_health = max(0, self.base_health)
        return kill_reward, base_penalty

    def _update_particles(self):
        for p in self.particles[:]:
            p.update()
            if p.is_dead():
                self.particles.remove(p)

    def _create_particles(self, pos, count, color, life, min_speed=1, max_speed=3):
        for _ in range(count):
            self.particles.append(Particle(pos, color, life, min_speed, max_speed))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "base_health": self.base_health,
            "money": self.money,
            "wave": self.wave_number,
        }

    def _render_game(self):
        # Draw path
        pygame.draw.lines(self.screen, self.COLOR_PATH, False, self.path_waypoints, 30)

        # Draw grid
        for r in range(self.GRID_ROWS + 1):
            y = r * self.GRID_BOX_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = c * self.GRID_BOX_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))

        # Draw base
        base_y = self.path_waypoints[-1][1]
        pygame.gfxdraw.box(self.screen, pygame.Rect(self.SCREEN_WIDTH-10, base_y-20, 10, 40), (*self.COLOR_BASE, 150))
        pygame.gfxdraw.rectangle(self.screen, pygame.Rect(self.SCREEN_WIDTH-10, base_y-20, 10, 40), self.COLOR_BASE)

        # Draw towers
        for tower in self.towers: tower.draw(self.screen)
        
        # Draw enemies
        for enemy in self.enemies: enemy.draw(self.screen)
        
        # Draw projectiles
        for p in self.projectiles: p.draw(self.screen)

        # Draw particles
        for p in self.particles: p.draw(self.screen)

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        rect = pygame.Rect(cursor_x * self.GRID_BOX_SIZE, cursor_y * self.GRID_BOX_SIZE, self.GRID_BOX_SIZE, self.GRID_BOX_SIZE)
        is_valid = self.grid[cursor_y][cursor_x] is None
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        pygame.gfxdraw.box(self.screen, rect, color)
    
    def _render_ui(self):
        # Health Bar
        health_ratio = self.base_health / 100
        bar_width = self.SCREEN_WIDTH // 3
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH_BG, (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, int(bar_width * health_ratio), 20))
        health_text = self.font_ui.render(f"BASE HP: {self.base_health}", True, self.COLOR_UI_TEXT)
        self.screen.blit(health_text, (15, 12))

        # Money
        money_text = self.font_ui.render(f"$ {self.money}", True, self.COLOR_MONEY)
        self.screen.blit(money_text, (bar_width + 30, 12))

        # Wave Info
        if self.wave_state == "INTERMISSION" and not self.win:
            wave_text = f"WAVE {self.wave_number+1} IN {self.intermission_timer//30 + 1}"
        else:
            wave_text = f"WAVE {self.wave_number} / {self.MAX_WAVES}"
        wave_surf = self.font_ui.render(wave_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_surf, (self.SCREEN_WIDTH - wave_surf.get_width() - 15, 12))

        # Game Over / Victory
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            text = "VICTORY!" if self.win else "GAME OVER"
            color = (100, 255, 100) if self.win else (255, 50, 50)
            text_surf = self.font_game_over.render(text, True, color)
            text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(text_surf, text_rect)

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

# --- Helper Classes ---

class Enemy:
    def __init__(self, path, health, speed, bounty):
        self.path = path
        self.path_index = 0
        self.pos = list(path[0])
        self.max_health = health
        self.health = health
        self.speed = speed
        self.bounty = bounty
        self.reached_end = False
        self.size = 8

    def move(self):
        if self.reached_end: return

        target_pos = self.path[self.path_index]
        direction = (target_pos[0] - self.pos[0], target_pos[1] - self.pos[1])
        dist = math.hypot(*direction)

        if dist < self.speed:
            self.pos = list(target_pos)
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.reached_end = True
        else:
            self.pos[0] += (direction[0] / dist) * self.speed
            self.pos[1] += (direction[1] / dist) * self.speed

    def draw(self, screen):
        # Body
        points = [
            (self.pos[0], self.pos[1] - self.size),
            (self.pos[0] - self.size * 0.866, self.pos[1] + self.size * 0.5),
            (self.pos[0] + self.size * 0.866, self.pos[1] + self.size * 0.5)
        ]
        pygame.gfxdraw.aapolygon(screen, points, (255, 80, 80))
        pygame.gfxdraw.filled_polygon(screen, points, (220, 50, 50))
        
        # Health bar
        bar_width = 20
        bar_height = 4
        health_ratio = self.health / self.max_health
        bg_rect = pygame.Rect(self.pos[0] - bar_width/2, self.pos[1] - self.size - 10, bar_width, bar_height)
        hp_rect = pygame.Rect(self.pos[0] - bar_width/2, self.pos[1] - self.size - 10, int(bar_width * health_ratio), bar_height)
        pygame.draw.rect(screen, (80, 0, 0), bg_rect)
        pygame.draw.rect(screen, (0, 255, 0), hp_rect)

class Tower:
    def __init__(self, x, y, range, cooldown, damage):
        self.pos = (x, y)
        self.range = range
        self.max_cooldown = cooldown
        self.cooldown = 0
        self.damage = damage

    def find_target(self, enemies):
        for enemy in enemies:
            dist = math.hypot(self.pos[0] - enemy.pos[0], self.pos[1] - enemy.pos[1])
            if dist <= self.range:
                return enemy
        return None

class TowerBasic(Tower):
    def __init__(self, x, y):
        super().__init__(x, y, GameEnv.TOWER_BASIC_RANGE, GameEnv.TOWER_BASIC_COOLDOWN, GameEnv.TOWER_BASIC_DMG)
        self.color = (0, 200, 255)
        self.proj_color = (150, 220, 255)

    def draw(self, screen):
        pygame.gfxdraw.aacircle(screen, int(self.pos[0]), int(self.pos[1]), 10, self.color)
        pygame.gfxdraw.filled_circle(screen, int(self.pos[0]), int(self.pos[1]), 10, (*self.color, 100))
        pygame.gfxdraw.filled_circle(screen, int(self.pos[0]), int(self.pos[1]), 5, self.color)

class TowerAdvanced(Tower):
    def __init__(self, x, y):
        super().__init__(x, y, GameEnv.TOWER_ADV_RANGE, GameEnv.TOWER_ADV_COOLDOWN, GameEnv.TOWER_ADV_DMG)
        self.color = (255, 150, 0)
        self.proj_color = (255, 200, 100)

    def draw(self, screen):
        rect = pygame.Rect(self.pos[0] - 8, self.pos[1] - 8, 16, 16)
        pygame.gfxdraw.box(screen, rect, (*self.color, 100))
        pygame.gfxdraw.rectangle(screen, rect, self.color)
        inner_rect = pygame.Rect(self.pos[0] - 4, self.pos[1] - 4, 8, 8)
        pygame.gfxdraw.box(screen, inner_rect, self.color)

class Projectile:
    def __init__(self, start_pos, target, damage, color):
        self.pos = list(start_pos)
        self.target = target
        self.damage = damage
        self.color = color
        self.speed = 10

    def move(self):
        direction = (self.target.pos[0] - self.pos[0], self.target.pos[1] - self.pos[1])
        dist = math.hypot(*direction)
        if dist == 0: return
        self.pos[0] += (direction[0] / dist) * self.speed
        self.pos[1] += (direction[1] / dist) * self.speed

    def check_collision(self):
        dist = math.hypot(self.pos[0] - self.target.pos[0], self.pos[1] - self.target.pos[1])
        return dist < self.target.size
    
    def is_out_of_bounds(self):
        return not (0 < self.pos[0] < GameEnv.SCREEN_WIDTH and 0 < self.pos[1] < GameEnv.SCREEN_HEIGHT)

    def draw(self, screen):
        pygame.draw.line(screen, self.color, self.pos, (self.pos[0]-self.speed/2, self.pos[1]-self.speed/2), 2)

class Particle:
    def __init__(self, pos, color, life, min_speed, max_speed):
        self.pos = list(pos)
        self.color = color
        self.max_life = life
        self.life = life
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(min_speed, max_speed)
        self.vel = [math.cos(angle) * speed, math.sin(angle) * speed]

    def update(self):
        self.life -= 1
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        self.vel[0] *= 0.95
        self.vel[1] *= 0.95

    def is_dead(self):
        return self.life <= 0

    def draw(self, screen):
        if self.is_dead(): return
        alpha = int(255 * (self.life / self.max_life))
        radius = int(3 * (self.life / self.max_life))
        if radius < 1: return
        
        temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
        screen.blit(temp_surf, (int(self.pos[0] - radius), int(self.pos[1] - radius)), special_flags=pygame.BLEND_RGBA_ADD)