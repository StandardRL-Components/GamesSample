import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# --- Helper Classes ---

class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, size, life, dx, dy):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life
        self.dx = dx
        self.dy = dy

    def update(self):
        self.x += self.dx
        self.y += self.dy
        self.life -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            color = self.color + (alpha,)
            try:
                pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), int(self.size), color)
            except OverflowError: # size can become too large briefly
                pass


class Projectile:
    """A projectile fired by a turret."""
    def __init__(self, x, y, target):
        self.x = x
        self.y = y
        self.target = target
        self.speed = 10
        self.radius = 3
        self.color = (255, 255, 0)
        self.alive = True

    def update(self):
        if not self.target.alive:
            self.alive = False
            return False

        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist = math.hypot(dx, dy)

        if dist < self.speed:
            self.alive = False
            return True  # Hit target
        
        self.x += (dx / dist) * self.speed
        self.y += (dy / dist) * self.speed
        return False

    def draw(self, surface):
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.radius, self.color)


class Block:
    """Base class for all blocks."""
    def __init__(self, grid_x, grid_y, cell_size):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.cell_size = cell_size
        self.rect = pygame.Rect(grid_x * cell_size, grid_y * cell_size, cell_size, cell_size)

    def update(self, *args, **kwargs):
        pass

    def draw(self, surface):
        pass


class WallBlock(Block):
    """A simple wall that obstructs enemies."""
    def __init__(self, grid_x, grid_y, cell_size):
        super().__init__(grid_x, grid_y, cell_size)
        self.color = (60, 120, 220)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect.inflate(-4, -4), border_radius=4)
        pygame.draw.rect(surface, (120, 180, 255), self.rect.inflate(-4, -4), width=2, border_radius=4)


class TurretBlock(Block):
    """A turret that automatically fires at enemies."""
    def __init__(self, grid_x, grid_y, cell_size):
        super().__init__(grid_x, grid_y, cell_size)
        self.color = (180, 40, 220)
        self.range = 100
        self.cooldown = 30  # steps
        self.cooldown_timer = 0
        self.center_x = self.rect.centerx
        self.center_y = self.rect.centery

    def update(self, enemies, projectiles):
        self.cooldown_timer = max(0, self.cooldown_timer - 1)
        if self.cooldown_timer == 0:
            target = self.find_target(enemies)
            if target:
                projectiles.append(Projectile(self.center_x, self.center_y, target))
                self.cooldown_timer = self.cooldown
                # sfx: turret_fire.wav

    def find_target(self, enemies):
        for enemy in enemies:
            dist = math.hypot(enemy.x - self.center_x, enemy.y - self.center_y)
            if dist <= self.range:
                return enemy
        return None

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect.inflate(-6, -6), border_radius=6)
        pygame.draw.circle(surface, (220, 100, 255), self.rect.center, 5)


class Enemy:
    """An enemy that moves towards the base."""
    def __init__(self, x, y, speed, health, cell_size, base_grid_pos):
        self.x = x
        self.y = y
        self.start_speed = speed
        self.speed = speed
        self.health = health
        self.max_health = health
        self.cell_size = cell_size
        self.base_grid_pos = base_grid_pos
        self.radius = int(cell_size * 0.4)
        self.color = (220, 50, 50)
        self.alive = True
        self.path = deque()

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            self.alive = False
            return True # Died
        return False

    def update(self, grid):
        target_x = (self.base_grid_pos[0] + 0.5) * self.cell_size
        target_y = (self.base_grid_pos[1] + 0.5) * self.cell_size

        grid_x, grid_y = int(self.x / self.cell_size), int(self.y / self.cell_size)

        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)
        if dist == 0: return

        move_x = (dx / dist) * self.speed
        move_y = (dy / dist) * self.speed
        
        next_x = self.x + move_x
        next_y = self.y + move_y
        
        next_grid_x = int(next_x / self.cell_size)
        next_grid_y = int(next_y / self.cell_size)

        # Simple collision with blocks
        if (next_grid_x, grid_y) in grid and grid_x != next_grid_x:
            move_x = 0
        if (grid_x, next_grid_y) in grid and grid_y != next_grid_y:
            move_y = 0

        self.x += move_x
        self.y += move_y

    def draw(self, surface):
        pygame.gfxdraw.filled_circle(surface, int(self.x), int(self.y), self.radius, self.color)
        pygame.gfxdraw.aacircle(surface, int(self.x), int(self.y), self.radius, self.color)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to place a block. "
        "Hold shift to cycle through available block types (Wall, Turret)."
    )

    game_description = (
        "A top-down tower defense game. Defend your base against waves of enemies "
        "by strategically placing walls and auto-firing turrets."
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 32, 20
        self.CELL_SIZE = self.WIDTH // self.GRID_W
        self.MAX_STEPS = 2500
        self.MAX_WAVES = 10

        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (30, 35, 40)
        self.COLOR_BASE = (50, 200, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        self.block_types = [WallBlock, TurretBlock]
        self.block_names = ["WALL", "TURRET"]

        # self.reset() is called by the wrapper, no need to call it here.
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False

        self.base_grid_pos = (self.GRID_W // 2, self.GRID_H // 2)
        self.base_health = 100
        self.base_rect = pygame.Rect(
            self.base_grid_pos[0] * self.CELL_SIZE,
            self.base_grid_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )

        self.wave_number = 0
        self.wave_in_progress = False
        self.wave_transition_timer = 90 # 3 seconds at 30fps

        self.enemies = []
        self.projectiles = []
        self.particles = []
        self.grid = {} # (x, y) -> Block object

        self.cursor_pos = [self.GRID_W // 2 - 3, self.GRID_H // 2]
        self.selected_block_type = 0
        self.last_space_held = False
        self.last_shift_held = False
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(30)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        
        self._handle_input(action)

        reward += self._update_game_state()
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated:
            if self.game_won:
                reward += 100
            else:
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1
        elif movement == 2: self.cursor_pos[1] += 1
        elif movement == 3: self.cursor_pos[0] -= 1
        elif movement == 4: self.cursor_pos[0] += 1
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_W - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_H - 1)

        # --- Place Block ---
        space_press = space_held and not self.last_space_held
        if space_press:
            pos = tuple(self.cursor_pos)
            if pos not in self.grid and pos != self.base_grid_pos:
                BlockClass = self.block_types[self.selected_block_type]
                self.grid[pos] = BlockClass(pos[0], pos[1], self.CELL_SIZE)
                # sfx: place_block.wav
                self._spawn_particles(
                    (pos[0] + 0.5) * self.CELL_SIZE, (pos[1] + 0.5) * self.CELL_SIZE,
                    (200, 200, 255), 15, size=10, spread=10
                )

        # --- Cycle Block Type ---
        shift_press = shift_held and not self.last_shift_held
        if shift_press:
            self.selected_block_type = (self.selected_block_type + 1) % len(self.block_types)
            # sfx: cycle_type.wav
        
        self.last_space_held = space_held
        self.last_shift_held = shift_held

    def _update_game_state(self):
        reward = 0

        # --- Wave Management ---
        if not self.wave_in_progress:
            self.wave_transition_timer -= 1
            if self.wave_transition_timer <= 0:
                self._start_new_wave()
        elif not self.enemies:
            reward += 1.0 # Wave survived reward
            self.wave_in_progress = False
            self.wave_transition_timer = 150 # 5 seconds for next wave
            if self.wave_number >= self.MAX_WAVES:
                self.game_won = True
                self.game_over = True

        # --- Update Blocks (Turrets) ---
        for block in self.grid.values():
            block.update(enemies=self.enemies, projectiles=self.projectiles)

        # --- Update Projectiles ---
        for p in self.projectiles[:]:
            hit = p.update()
            if not p.alive or hit:
                self.projectiles.remove(p)
                if hit and p.target.alive:
                    if p.target.take_damage(1):
                        # sfx: enemy_die.wav
                        reward += 0.1 # Enemy destroyed reward
                        self.score += 10
                        self._spawn_particles(p.target.x, p.target.y, (255, 80, 80), 20, size=20)
                    else:
                        # sfx: enemy_hit.wav
                        self._spawn_particles(p.target.x, p.target.y, (255, 255, 0), 5, size=5)

        # --- Update Enemies ---
        for enemy in self.enemies[:]:
            if not enemy.alive:
                self.enemies.remove(enemy)
                continue
            
            enemy.update(self.grid)
            
            # Check collision with base
            if self.base_rect.collidepoint(enemy.x, enemy.y):
                self.base_health -= 1
                self.score -= 1 # Small penalty
                enemy.alive = False
                reward -= 0.5 # Base hit penalty
                # sfx: base_hit.wav
                self._spawn_particles(enemy.x, enemy.y, self.COLOR_BASE, 10, size=15)

        # --- Update Particles ---
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)
        
        return reward

    def _start_new_wave(self):
        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES: return

        self.wave_in_progress = True
        num_enemies = 5 + self.wave_number * 2
        enemy_speed = 0.5 + (self.wave_number - 1) * 0.1
        enemy_health = 1 + self.wave_number // 3
        
        for _ in range(num_enemies):
            side = self.np_random.integers(4)
            if side == 0: # Top
                x, y = self.np_random.uniform(0, self.WIDTH), -self.CELL_SIZE
            elif side == 1: # Bottom
                x, y = self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.CELL_SIZE
            elif side == 2: # Left
                x, y = -self.CELL_SIZE, self.np_random.uniform(0, self.HEIGHT)
            else: # Right
                x, y = self.WIDTH + self.CELL_SIZE, self.np_random.uniform(0, self.HEIGHT)
            
            self.enemies.append(Enemy(x, y, enemy_speed, enemy_health, self.CELL_SIZE, self.base_grid_pos))

    def _check_termination(self):
        if self.base_health <= 0:
            self.game_over = True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
        return self.game_over

    def _spawn_particles(self, x, y, color, count, size=5, life=30, spread=5):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, spread)
            dx = math.cos(angle) * speed
            dy = math.sin(angle) * speed
            self.particles.append(Particle(x, y, color, size, life, dx, dy))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Grid
        for x in range(0, self.WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.HEIGHT))
        for y in range(0, self.HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.WIDTH, y))
        
        # Base
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect, border_radius=4)
        pygame.draw.rect(self.screen, (150, 255, 150), self.base_rect, width=2, border_radius=4)

        # Blocks
        for block in self.grid.values():
            block.draw(self.screen)

        # Enemies
        for enemy in self.enemies:
            enemy.draw(self.screen)

        # Projectiles
        for p in self.projectiles:
            p.draw(self.screen)

        # Particles
        for p in self.particles:
            p.draw(self.screen)

        # Cursor
        if not self.game_over:
            cursor_rect = pygame.Rect(
                self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            is_valid_pos = tuple(self.cursor_pos) not in self.grid and tuple(self.cursor_pos) != self.base_grid_pos
            cursor_color = (255, 255, 255) if is_valid_pos else (255, 0, 0)
            
            # Pulsing alpha
            alpha = 100 + 50 * math.sin(self.steps * 0.2)
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((*cursor_color, alpha))
            self.screen.blit(s, cursor_rect.topleft)
            pygame.draw.rect(self.screen, cursor_color, cursor_rect, 2)
    
    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave_number}/{self.MAX_WAVES}", True, (255, 255, 255))
        self.screen.blit(wave_text, (10, 10))
        
        # Selected Block
        block_name = self.block_names[self.selected_block_type]
        block_text = self.font_small.render(f"SELECT: {block_name}", True, (255, 255, 255))
        self.screen.blit(block_text, (self.WIDTH // 2 - block_text.get_width() // 2, 10))

        # Health Bar
        health_bar_width = 200
        health_ratio = max(0, self.base_health / 100)
        health_bar_bg = pygame.Rect((self.WIDTH - health_bar_width) // 2, self.HEIGHT - 30, health_bar_width, 20)
        health_bar_fg = pygame.Rect(health_bar_bg.left, health_bar_bg.top, int(health_bar_width * health_ratio), 20)
        pygame.draw.rect(self.screen, (80, 0, 0), health_bar_bg, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BASE, health_bar_fg, border_radius=5)
        health_text = self.font_small.render(f"BASE HEALTH", True, (255, 255, 255))
        self.screen.blit(health_text, (health_bar_bg.centerx - health_text.get_width() // 2, health_bar_bg.y - 18))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WON!" if self.game_won else "GAME OVER"
            color = (100, 255, 100) if self.game_won else (255, 100, 100)
            end_text = self.font_large.render(message, True, color)
            self.screen.blit(end_text, (self.WIDTH // 2 - end_text.get_width() // 2, self.HEIGHT // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave_number,
            "base_health": self.base_health,
        }
    
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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