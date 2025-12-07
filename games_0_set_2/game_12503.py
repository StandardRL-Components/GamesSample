import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T18:00:09.129449
# Source Brief: brief_02503.md
# Brief Index: 2503
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

# Helper classes for game entities
class Cell:
    def __init__(self, x, y, level, cell_size):
        self.x = x
        self.y = y
        self.level = level
        self.max_health = 10 * self.level
        self.health = self.max_health
        self.cell_size = cell_size
        self.is_settled = False
        self.division_timer = 200  # Ticks until division
        self.target_y = y
        self.visual_y = y * cell_size

    def damage(self, amount):
        self.health -= amount
        return self.health <= 0

    def draw(self, screen, colors, font):
        # Interpolate visual position for smooth falling
        self.visual_y += (self.y * self.cell_size - self.visual_y) * 0.2
        
        px, py = int(self.x * self.cell_size + self.cell_size / 2), int(self.visual_y + self.cell_size / 2)
        radius = int(self.cell_size / 2 * 0.85)
        color_index = min(self.level - 1, len(colors) - 1)
        color = colors[color_index]

        # Glow effect for higher levels
        if self.level > 2:
            glow_radius = int(radius * (1.2 + self.level * 0.05))
            glow_color = color + (60,)  # Add alpha
            temp_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, glow_color, (glow_radius, glow_radius), glow_radius)
            screen.blit(temp_surf, (px - glow_radius, py - glow_radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Main cell body
        pygame.gfxdraw.filled_circle(screen, px, py, radius, color)
        pygame.gfxdraw.aacircle(screen, px, py, radius, (200, 200, 220))

        # Health bar
        if self.health < self.max_health:
            bar_width = self.cell_size * 0.8
            bar_height = 4
            bar_x = px - bar_width / 2
            bar_y = py + radius + 2
            health_ratio = self.health / self.max_health
            pygame.draw.rect(screen, (50, 0, 0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, bar_width * health_ratio, bar_height))

        # Level text
        level_text = font.render(str(self.level), True, (255, 255, 255))
        text_rect = level_text.get_rect(center=(px, py))
        screen.blit(level_text, text_rect)

class Enemy:
    def __init__(self, x, y, health, speed, damage, color):
        self.x = x
        self.y = y
        self.health = health
        self.max_health = health
        self.speed = speed
        self.damage = damage
        self.color = color
        self.size = 12

    def move(self, base_y):
        if self.y < base_y:
            self.y += self.speed
        else:
            self.y -= self.speed

    def draw(self, screen):
        px, py = int(self.x), int(self.y)
        points = [
            (px, py - self.size),
            (px + self.size // 2, py + self.size // 2),
            (px - self.size // 2, py + self.size // 2),
        ]
        pygame.gfxdraw.aapolygon(screen, points, self.color)
        pygame.gfxdraw.filled_polygon(screen, points, self.color)
        
        # Simple health indicator
        if self.health < self.max_health:
            ratio = self.health / self.max_health
            bright_color = tuple(min(255, int(c * ratio + (255-c)*0.2)) for c in self.color)
            pygame.gfxdraw.filled_polygon(screen, points, bright_color)


class Particle:
    def __init__(self, x, y, color, lifetime):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = lifetime

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.98
        self.vy *= 0.98
        self.lifetime -= 1
        return self.lifetime <= 0

    def draw(self, screen):
        alpha = int(255 * (self.lifetime / self.max_lifetime))
        color = self.color + (alpha,)
        size = int(3 * (self.lifetime / self.max_lifetime))
        if size > 0:
            temp_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (size, size), size)
            screen.blit(temp_surf, (int(self.x - size), int(self.y - size)))

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "Defend your base by merging and evolving cells. Flip gravity to rearrange your defenses and call waves of enemies to challenge your setup."
    )
    user_guide = (
        "Press space to flip gravity. Submit a no-op action (e.g., no keys pressed) to advance to the next enemy wave."
    )
    auto_advance = False

    # --- CONSTANTS ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 5
    CELL_SIZE = 40
    ARENA_X_START = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    
    COLOR_BG = (15, 20, 35)
    COLOR_GRID = (30, 40, 60)
    COLOR_BASE = (60, 120, 200)
    COLOR_ENEMY = (255, 50, 50)
    CELL_COLORS = [
        (40, 180, 120), (60, 200, 140), (80, 220, 160),
        (100, 240, 180), (150, 255, 200), (200, 255, 220)
    ]
    COLOR_GRAVITY_UP = (255, 220, 100)
    COLOR_GRAVITY_DOWN = (255, 180, 50)
    COLOR_UI_TEXT = (220, 220, 240)

    MAX_STEPS = 3000
    MAX_WAVES = 20
    BASE_START_HEALTH = 100

    REWARD_ENEMY_DESTROYED = 1.0
    REWARD_BASE_HEALTH_LOST = -0.1
    REWARD_WAVE_CLEAR = 5.0
    REWARD_CELL_LEVEL_UP = 2.0
    REWARD_WIN = 100.0
    REWARD_LOSS = -100.0
    REWARD_INVALID_WAVE_ADVANCE = -0.5

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
        self.cell_font = pygame.font.Font(None, 20)

        self.cells = []
        self.enemies = []
        self.particles = []
        
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.wave_number = 0
        self.base_health = self.BASE_START_HEALTH
        self.gravity_direction = 1  # 1 for down, -1 for up
        
        self.base_y = self.SCREEN_HEIGHT - self.CELL_SIZE * self.GRID_HEIGHT - 20
        self.base_rect = pygame.Rect(self.ARENA_X_START, self.base_y, self.GRID_WIDTH * self.CELL_SIZE, 10)

        self.cells = []
        self.enemies = []
        self.particles = []
        
        self.prev_space_held = False
        self.wave_in_progress = False

        self._spawn_initial_cells()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Player Actions ---
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held

        if space_pressed:
            self._flip_gravity()
        elif movement == 0:
            reward += self._advance_wave()
        
        # --- Update Game State ---
        reward += self._update_cells()
        self._update_enemies()
        reward += self._handle_combat()
        reward += self._handle_matches()
        self._update_particles()
        
        # Check for wave clear
        if self.wave_in_progress and not self.enemies:
            self.wave_in_progress = False
            reward += self.REWARD_WAVE_CLEAR
            self.score += self.REWARD_WAVE_CLEAR # Update score for UI
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated:
            if self.base_health <= 0:
                reward += self.REWARD_LOSS
            elif self.wave_number > self.MAX_WAVES:
                reward += self.REWARD_WIN
        
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _flip_gravity(self):
        # sound: 'gravity_shift.wav'
        self.gravity_direction *= -1
        for cell in self.cells:
            cell.is_settled = False
        self._create_particles(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, (255, 255, 100), 30)

    def _advance_wave(self):
        if self.wave_in_progress:
            return self.REWARD_INVALID_WAVE_ADVANCE

        self.wave_number += 1
        if self.wave_number > self.MAX_WAVES:
            return 0
            
        self.wave_in_progress = True
        
        num_enemies = 2 + self.wave_number // 2
        enemy_speed = 0.5 + self.wave_number * 0.1
        enemy_health = 10 + self.wave_number * 5
        enemy_damage = 5 + self.wave_number

        for _ in range(num_enemies):
            spawn_x = self.ARENA_X_START + random.uniform(0, self.GRID_WIDTH * self.CELL_SIZE)
            spawn_y = 0 if self.gravity_direction == 1 else self.SCREEN_HEIGHT
            self.enemies.append(Enemy(spawn_x, spawn_y, enemy_health, enemy_speed, enemy_damage, self.COLOR_ENEMY))
        
        # sound: 'new_wave.wav'
        return 0

    def _update_cells(self):
        reward = 0
        grid_occupancy = {}
        for cell in self.cells:
            if cell.is_settled:
                grid_occupancy[(cell.x, cell.y)] = cell
                cell.division_timer -= 1
                if cell.division_timer <= 0:
                    # sound: 'cell_divide.wav'
                    self._spawn_cell(cell.x, cell.y - self.gravity_direction, 1)
                    cell.division_timer = 200 + random.randint(-20, 20)

        for cell in self.cells:
            if not cell.is_settled:
                target_y = (self.GRID_HEIGHT - 1) if self.gravity_direction == 1 else 0
                
                # Check for cells below
                for y_check in range(cell.y + self.gravity_direction, self.GRID_HEIGHT if self.gravity_direction == 1 else -1, self.gravity_direction):
                    if (cell.x, y_check) in grid_occupancy:
                        target_y = y_check - self.gravity_direction
                        break
                
                if cell.y != target_y:
                    cell.y = target_y
                else:
                    cell.is_settled = True
                    grid_occupancy[(cell.x, cell.y)] = cell
        return reward

    def _update_enemies(self):
        base_target_y = self.base_rect.centery
        for enemy in self.enemies:
            enemy.move(base_target_y)

    def _handle_combat(self):
        reward = 0
        enemies_to_remove = []
        cells_to_remove = []

        for enemy in self.enemies:
            collided_cell = None
            for cell in self.cells:
                cell_rect = pygame.Rect(self.ARENA_X_START + cell.x * self.CELL_SIZE, cell.visual_y, self.CELL_SIZE, self.CELL_SIZE)
                enemy_rect = pygame.Rect(enemy.x - enemy.size/2, enemy.y - enemy.size/2, enemy.size, enemy.size)
                if cell_rect.colliderect(enemy_rect):
                    collided_cell = cell
                    break
            
            if collided_cell:
                # sound: 'impact.wav'
                if collided_cell.damage(enemy.damage):
                    if collided_cell not in cells_to_remove:
                        cells_to_remove.append(collided_cell)
                    self._create_particles(self.ARENA_X_START + collided_cell.x * self.CELL_SIZE + self.CELL_SIZE/2, collided_cell.visual_y + self.CELL_SIZE/2, self.CELL_COLORS[0], 20)
                
                enemy_damage_dealt = collided_cell.level * 5
                enemy.health -= enemy_damage_dealt
                if enemy.health <= 0:
                    if enemy not in enemies_to_remove:
                        # sound: 'enemy_die.wav'
                        enemies_to_remove.append(enemy)
                        reward += self.REWARD_ENEMY_DESTROYED
                        self._create_particles(enemy.x, enemy.y, self.COLOR_ENEMY, 40)
            
            # Check for base collision
            elif abs(enemy.y - self.base_rect.centery) < enemy.speed + 2:
                if self.base_rect.left < enemy.x < self.base_rect.right:
                    # sound: 'base_damage.wav'
                    self.base_health -= enemy.damage
                    reward += self.REWARD_BASE_HEALTH_LOST * enemy.damage
                    if enemy not in enemies_to_remove:
                        enemies_to_remove.append(enemy)
                    self._create_particles(enemy.x, enemy.y, self.COLOR_BASE, 30)

        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        self.cells = [c for c in self.cells if c not in cells_to_remove]
        return reward

    def _handle_matches(self):
        reward = 0
        grid = {}
        for cell in self.cells:
            if cell.is_settled:
                grid[(cell.x, cell.y)] = cell
        
        matched_cells = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH - 1):
                cell1 = grid.get((x, y))
                cell2 = grid.get((x + 1, y))
                
                if cell1 and cell2 and cell1.level == cell2.level and cell1 not in matched_cells and cell2 not in matched_cells:
                    # sound: 'match.wav'
                    matched_cells.add(cell1)
                    matched_cells.add(cell2)
                    
                    new_level = cell1.level + 1
                    new_cell = Cell(x, y, new_level, self.CELL_SIZE)
                    new_cell.is_settled = False # It needs to fall into place
                    
                    self.cells.append(new_cell)
                    reward += self.REWARD_CELL_LEVEL_UP * new_level
                    
                    px = self.ARENA_X_START + (x + 0.5) * self.CELL_SIZE
                    py = y * self.CELL_SIZE + self.base_rect.top
                    color_index = min(new_level - 1, len(self.CELL_COLORS) - 1)
                    self._create_particles(px, py, self.CELL_COLORS[color_index], 50)
        
        if matched_cells:
            self.cells = [c for c in self.cells if c not in matched_cells]
            # Unsettle cells above the matched ones
            for mc in matched_cells:
                for cell in self.cells:
                    if cell.x == mc.x and ((self.gravity_direction == 1 and cell.y < mc.y) or (self.gravity_direction == -1 and cell.y > mc.y)):
                        cell.is_settled = False

        return reward

    def _update_particles(self):
        self.particles = [p for p in self.particles if not p.update()]

    def _spawn_cell(self, grid_x, grid_y, level):
        if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
            # Check if position is occupied
            is_occupied = any(c.x == grid_x and c.y == grid_y for c in self.cells)
            if not is_occupied:
                self.cells.append(Cell(grid_x, grid_y, level, self.CELL_SIZE))

    def _spawn_initial_cells(self):
        for _ in range(5):
            x = self.np_random.integers(0, self.GRID_WIDTH)
            y = self.np_random.integers(self.GRID_HEIGHT - 2, self.GRID_HEIGHT)
            self._spawn_cell(x, y, 1)

    def _create_particles(self, x, y, color, count):
        for _ in range(count):
            self.particles.append(Particle(x, y, color, random.randint(20, 40)))

    def _check_termination(self):
        if self.game_over: return True
        self.game_over = (self.base_health <= 0 or 
                          self.wave_number > self.MAX_WAVES or
                          self.steps >= self.MAX_STEPS)
        return self.game_over

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave_number, "base_health": self.base_health}

    def _render_game(self):
        self._draw_grid()
        self._draw_base()
        for cell in self.cells:
            cell.draw(self.screen, self.CELL_COLORS, self.cell_font)
        for enemy in self.enemies:
            enemy.draw(self.screen)
        for particle in self.particles:
            particle.draw(self.screen)

    def _draw_grid(self):
        arena_top = self.base_rect.top
        for x in range(self.GRID_WIDTH + 1):
            px = self.ARENA_X_START + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, arena_top), (px, arena_top + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = arena_top + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.ARENA_X_START, py), (self.ARENA_X_START + self.GRID_WIDTH * self.CELL_SIZE, py))

    def _draw_base(self):
        pygame.draw.rect(self.screen, self.COLOR_BASE, self.base_rect, border_radius=3)
        # Health bar for the base
        health_ratio = max(0, self.base_health / self.BASE_START_HEALTH)
        health_color = (int(255 * (1 - health_ratio)), int(255 * health_ratio), 0)
        health_width = self.base_rect.width * health_ratio
        health_rect = pygame.Rect(self.base_rect.left, self.base_rect.top, health_width, self.base_rect.height)
        pygame.draw.rect(self.screen, health_color, health_rect, border_radius=3)

    def _render_ui(self):
        # Score
        score_text = self.font_medium.render(f"Score: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_large.render(f"Wave: {self.wave_number}/{self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        wave_rect = wave_text.get_rect(centerx=self.SCREEN_WIDTH / 2, y=10)
        self.screen.blit(wave_text, wave_rect)

        # Base Health
        health_text = self.font_medium.render(f"Base Health: {int(self.base_health)}", True, self.COLOR_UI_TEXT)
        health_rect = health_text.get_rect(right=self.SCREEN_WIDTH - 10, y=10)
        self.screen.blit(health_text, health_rect)

        # Gravity Indicator
        arrow_y = self.SCREEN_HEIGHT / 2
        arrow_color = self.COLOR_GRAVITY_DOWN if self.gravity_direction == 1 else self.COLOR_GRAVITY_UP
        if self.gravity_direction == 1:
            points = [(20, arrow_y - 10), (40, arrow_y - 10), (30, arrow_y + 10)]
        else:
            points = [(30, arrow_y - 10), (20, arrow_y + 10), (40, arrow_y + 10)]
        pygame.gfxdraw.aapolygon(self.screen, points, arrow_color)
        pygame.gfxdraw.filled_polygon(self.screen, points, arrow_color)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            msg = "VICTORY!" if self.wave_number > self.MAX_WAVES else "GAME OVER"
            end_text = self.font_large.render(msg, True, (255, 255, 255))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, end_rect)


if __name__ == '__main__':
    # Example of how to run the environment
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    total_reward = 0
    
    # Replace Pygame screen with a display screen for human play
    pygame.display.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cellular Defense")
    env.screen = screen # a bit of a hack to reuse rendering logic

    while running:
        action = [0, 0, 0] # Default: no-op (advance wave)
        
        # Check for held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            action[1] = 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q: # Quit
                    running = False
                if event.key == pygame.K_r: # Reset
                    obs, info = env.reset()
                    total_reward = 0
        
        # In human play, we only take an action if a key was pressed or space is held
        # The main logic is driven by the clock tick, but the env state only changes on step()
        # For this game, `auto_advance=False` means we need to call step() to see changes.
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        pygame.surfarray.blit_array(screen, np.transpose(obs, (1, 0, 2)))
        pygame.display.flip()

        env.clock.tick(30) # Run at 30 FPS for human play

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

    pygame.quit()