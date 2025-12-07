# Generated: 2025-08-28T05:50:51.231364
# Source Brief: brief_05694.md
# Brief Index: 5694

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque, namedtuple
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Helper classes for game objects
class Block:
    def __init__(self, grid_pos, block_type, properties):
        self.grid_pos = grid_pos
        self.type = block_type
        self.max_health = properties.get('health', 0)
        self.health = self.max_health
        self.is_wall = properties.get('is_wall', False)
        self.color = properties.get('color', (255, 255, 255))


class Enemy:
    def __init__(self, grid_pos, health, speed, damage, wave_bonus):
        self.grid_pos = list(grid_pos) # [x, y]
        self.pixel_pos = [0, 0] # For smooth animation
        self.health = health * (1 + wave_bonus)
        self.max_health = self.health
        self.speed = speed * (1 + wave_bonus)
        self.damage = damage
        self.path = deque()
        self.move_progress = 1.0 # Start ready to move
        self.attack_cooldown = 0

class Projectile:
    def __init__(self, start_pos, target_enemy, speed=15, damage=25):
        self.pos = list(start_pos)
        self.target = target_enemy
        self.speed = speed
        self.damage = damage
        self.terminated = False

class Particle:
    def __init__(self, x, y, color, life, size_range=(1, 4), speed_range=(-2, 2)):
        self.x = x
        self.y = y
        self.vx = random.uniform(*speed_range)
        self.vy = random.uniform(*speed_range)
        self.color = color
        self.life = life
        self.max_life = life
        self.size = random.uniform(*size_range)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place the selected block. "
        "Press Shift to cycle through available block types."
    )

    game_description = (
        "An isometric tower defense game. Strategically place blocks to build a fortress and defend "
        "it from waves of enemies. Survive all waves to win."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE_X, GRID_SIZE_Y = 16, 10
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = 20, 10
    MAX_STEPS = 30 * 60 * 2  # 2 minutes at 30fps

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (50, 60, 70)
    COLOR_FORTRESS = (100, 200, 255)
    COLOR_ENEMY = (255, 50, 50)
    COLOR_TEXT = (240, 240, 240)
    COLOR_HEALTH_GREEN = (50, 220, 50)
    COLOR_HEALTH_RED = (220, 50, 50)
    COLOR_CURSOR_VALID = (255, 255, 255, 150)
    COLOR_CURSOR_INVALID = (255, 50, 50, 150)
    
    BLOCK_PROPERTIES = {
        'BASIC': {'color': (100, 150, 100), 'health': 100, 'cost': 1},
        'WALL': {'color': (150, 160, 170), 'health': 500, 'cost': 2, 'is_wall': True},
        'CANNON': {'color': (200, 200, 50), 'health': 75, 'cost': 3, 'range': 5, 'fire_rate': 45, 'damage': 30},
    }
    BLOCK_TYPES = list(BLOCK_PROPERTIES.keys())

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
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)

        self.screen_center_x = self.SCREEN_WIDTH / 2
        self.screen_center_y = self.SCREEN_HEIGHT / 2 - (self.GRID_SIZE_Y * self.TILE_HEIGHT_HALF) / 2

        self.reset()
        # self.validate_implementation() # Commented out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed Python's random and NumPy's RNG
            random.seed(seed)
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.reward_this_step = 0

        self.fortress_pos = (self.GRID_SIZE_X - 1, self.GRID_SIZE_Y // 2)
        self.fortress_health = 100
        self.fortress_max_health = 100

        self.grid = [[None for _ in range(self.GRID_SIZE_Y)] for _ in range(self.GRID_SIZE_X)]
        self.blocks = []
        self.enemies = []
        self.projectiles = []
        self.particles = []

        self.current_wave = 1
        self.game_phase = 'BUILDING_START'
        self.phase_timer = 90  # 3 seconds for message

        self.cursor_pos = [self.GRID_SIZE_X // 2, self.GRID_SIZE_Y // 2]
        self.selected_block_idx = 0
        self._replenish_inventory()

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.reward_this_step = 0
        
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        self.prev_space_held, self.prev_shift_held = space_held, shift_held

        if not self.game_over:
            self._update_game_phase(movement, space_press, shift_press)
            self._update_particles()
        
        self.steps += 1
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_phase(self, movement, space_press, shift_press):
        if self.game_phase == 'BUILDING_START':
            self.phase_timer -= 1
            if self.phase_timer <= 0:
                self.game_phase = 'BUILDING'
                self.phase_timer = 30 * 10 # 10 seconds build time

        elif self.game_phase == 'BUILDING':
            self.phase_timer -= 1
            self._handle_cursor_movement(movement)
            if shift_press:
                self.selected_block_idx = (self.selected_block_idx + 1) % len(self.BLOCK_TYPES)
            if space_press:
                self._place_block()
            
            if self.phase_timer <= 0:
                self.game_phase = 'WAVE_START'

        elif self.game_phase == 'WAVE_START':
            self._spawn_enemies()
            self.game_phase = 'WAVE'

        elif self.game_phase == 'WAVE':
            self._update_cannons()
            self._update_projectiles()
            self._update_enemies()

            if not self.enemies and not self.game_over:
                self.reward_this_step += 1.0 # Wave survived
                self.current_wave += 1
                if self.current_wave > 20:
                    self.win = True
                    self.game_over = True
                    self.reward_this_step += 100.0
                else:
                    self.game_phase = 'BUILDING_START'
                    self.phase_timer = 90
                    self._replenish_inventory()

    def _check_termination(self):
        if self.game_over:
            return True
        if self.fortress_health <= 0:
            self.fortress_health = 0
            self.game_over = True
            self.reward_this_step = -100.0
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _handle_cursor_movement(self, movement):
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1 # Up
        elif movement == 2 and self.cursor_pos[1] < self.GRID_SIZE_Y - 1: self.cursor_pos[1] += 1 # Down
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1 # Left
        elif movement == 4 and self.cursor_pos[0] < self.GRID_SIZE_X - 1: self.cursor_pos[0] += 1 # Right

    def _place_block(self):
        x, y = self.cursor_pos
        block_type = self.BLOCK_TYPES[self.selected_block_idx]
        
        if self.grid[x][y] is None and self.block_inventory[block_type] > 0 and (x, y) != self.fortress_pos:
            properties = self.BLOCK_PROPERTIES[block_type]
            new_block = Block((x, y), block_type, properties)
            self.grid[x][y] = new_block
            self.blocks.append(new_block)
            self.block_inventory[block_type] -= 1
            self.reward_this_step -= 0.01
            # Invalidate enemy paths
            for enemy in self.enemies:
                enemy.path.clear()
            # sfx: place_block
            self._create_particles(self._grid_to_screen(x,y), (200,200,200), 10)

    def _replenish_inventory(self):
        self.block_inventory = {'BASIC': 5, 'WALL': 3, 'CANNON': 2}

    def _spawn_enemies(self):
        num_enemies = 2 + self.current_wave
        spawn_points = [(0, i) for i in range(self.GRID_SIZE_Y)]
        wave_bonus = (self.current_wave - 1) * 0.05
        
        for _ in range(num_enemies):
            spawn_pos = random.choice(spawn_points)
            if self.grid[spawn_pos[0]][spawn_pos[1]] is None:
                enemy = Enemy(spawn_pos, health=50, speed=0.02, damage=10, wave_bonus=wave_bonus)
                enemy.pixel_pos = list(self._grid_to_screen(*enemy.grid_pos))
                self.enemies.append(enemy)

    def _update_cannons(self):
        for block in self.blocks:
            if block.type == 'CANNON':
                # Cannon logic stored directly on block object for simplicity
                if not hasattr(block, 'cooldown'):
                    block.cooldown = 0
                
                block.cooldown = max(0, block.cooldown - 1)
                if block.cooldown == 0:
                    props = self.BLOCK_PROPERTIES['CANNON']
                    target = None
                    min_dist = float('inf')
                    for enemy in self.enemies:
                        dist = math.hypot(enemy.grid_pos[0] - block.grid_pos[0], enemy.grid_pos[1] - block.grid_pos[1])
                        if dist <= props['range'] and dist < min_dist:
                            min_dist = dist
                            target = enemy
                    
                    if target:
                        start_pos = self._grid_to_screen(*block.grid_pos)
                        start_pos = (start_pos[0], start_pos[1] - 15) # Fire from top of cannon
                        self.projectiles.append(Projectile(start_pos, target, damage=props['damage']))
                        block.cooldown = props['fire_rate']
                        # sfx: cannon_fire

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            if p.terminated or p.target not in self.enemies:
                if p in self.projectiles:
                    self.projectiles.remove(p)
                continue
            
            target_pos = p.target.pixel_pos
            dx = target_pos[0] - p.pos[0]
            dy = target_pos[1] - p.pos[1]
            dist = math.hypot(dx, dy)

            if dist < p.speed:
                p.target.health -= p.damage
                self._create_particles(p.pos, self.COLOR_ENEMY, 15)
                # sfx: projectile_hit
                if p.target.health <= 0:
                    self._create_particles(p.target.pixel_pos, self.COLOR_ENEMY, 40, size_range=(2,6))
                    if p.target in self.enemies:
                        self.enemies.remove(p.target)
                    self.reward_this_step += 0.1
                    # sfx: enemy_destroyed
                p.terminated = True
                if p in self.projectiles:
                    self.projectiles.remove(p)
            else:
                p.pos[0] += (dx / dist) * p.speed
                p.pos[1] += (dy / dist) * p.speed

    def _update_enemies(self):
        for enemy in self.enemies:
            if enemy.attack_cooldown > 0:
                enemy.attack_cooldown -= 1
                continue

            target_pos = self._find_path_target(enemy.grid_pos)

            if target_pos is None: # Blocked or at destination
                self._handle_enemy_attack(enemy)
                continue

            if enemy.move_progress >= 1.0:
                # Check if next step is blocked
                next_grid_pos = target_pos
                if self.grid[next_grid_pos[0]][next_grid_pos[1]] is not None or next_grid_pos == self.fortress_pos:
                     self._handle_enemy_attack(enemy)
                     continue

                enemy.grid_pos = next_grid_pos
                enemy.move_progress = 0.0
            
            start_pixel = self._grid_to_screen(enemy.grid_pos[0], enemy.grid_pos[1])
            target_pixel = self._grid_to_screen(target_pos[0], target_pos[1])
            
            enemy.move_progress = min(1.0, enemy.move_progress + enemy.speed)
            
            enemy.pixel_pos[0] = start_pixel[0] + (target_pixel[0] - start_pixel[0]) * enemy.move_progress
            enemy.pixel_pos[1] = start_pixel[1] + (target_pixel[1] - start_pixel[1]) * enemy.move_progress


    def _find_path_target(self, pos):
        gx, gy = int(pos[0]), int(pos[1])
        fx, fy = self.fortress_pos

        neighbors = [(gx + 1, gy), (gx - 1, gy), (gx, gy + 1), (gx, gy - 1)]
        valid_neighbors = []
        for nx, ny in neighbors:
            if 0 <= nx < self.GRID_SIZE_X and 0 <= ny < self.GRID_SIZE_Y:
                block = self.grid[nx][ny]
                if block is None or not block.is_wall:
                    valid_neighbors.append((nx, ny))

        if not valid_neighbors:
            return None

        best_neighbor = min(valid_neighbors, key=lambda p: math.hypot(p[0] - fx, p[1] - fy))
        return best_neighbor

    def _handle_enemy_attack(self, enemy):
        enemy.attack_cooldown = 60 # 2 seconds
        # Find adjacent target
        gx, gy = int(enemy.grid_pos[0]), int(enemy.grid_pos[1])
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = gx + dx, gy + dy
            if (nx, ny) == self.fortress_pos:
                self.fortress_health -= enemy.damage
                self._create_particles(self._grid_to_screen(nx,ny), self.COLOR_FORTRESS, 20)
                # sfx: fortress_hit
                return
            if 0 <= nx < self.GRID_SIZE_X and 0 <= ny < self.GRID_SIZE_Y:
                block = self.grid[nx][ny]
                if block is not None:
                    block.health -= enemy.damage
                    self._create_particles(self._grid_to_screen(nx,ny), block.color, 15)
                    # sfx: block_hit
                    if block.health <= 0:
                        self.grid[nx][ny] = None
                        if block in self.blocks:
                            self.blocks.remove(block)
                        self._create_particles(self._grid_to_screen(nx,ny), (128,128,128), 30)
                        # sfx: block_destroyed
                    return

    def _create_particles(self, pos, color, count, size_range=(1,4), speed_range=(-1.5,1.5)):
        for _ in range(count):
            self.particles.append(Particle(pos[0], pos[1], color, life=self.np_random.integers(20, 41), size_range=size_range, speed_range=speed_range))

    def _update_particles(self):
        for p in self.particles[:]:
            p.x += p.vx
            p.y += p.vy
            p.life -= 1
            if p.life <= 0:
                if p in self.particles:
                    self.particles.remove(p)

    def _grid_to_screen(self, grid_x, grid_y):
        screen_x = self.screen_center_x + (grid_x - grid_y) * self.TILE_WIDTH_HALF
        screen_y = self.screen_center_y + (grid_x + grid_y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_SIZE_Y + 1):
            start = self._grid_to_screen(-0.5, y - 0.5)
            end = self._grid_to_screen(self.GRID_SIZE_X - 0.5, y - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)
        for x in range(self.GRID_SIZE_X + 1):
            start = self._grid_to_screen(x - 0.5, -0.5)
            end = self._grid_to_screen(x - 0.5, self.GRID_SIZE_Y - 0.5)
            pygame.draw.aaline(self.screen, self.COLOR_GRID, start, end)

        # Collect all dynamic objects to sort for rendering
        render_queue = []
        for block in self.blocks:
            render_queue.append(('block', block))
        for enemy in self.enemies:
            render_queue.append(('enemy', enemy))
        
        # Add fortress to queue
        # The original dict caused an AttributeError in the sort key.
        # Create a simple object with a .grid_pos attribute to match other renderable items.
        Renderable = namedtuple('Renderable', ['grid_pos'])
        render_queue.append(('fortress', Renderable(grid_pos=self.fortress_pos)))

        # Sort by grid y then x for pseudo-3D effect
        render_queue.sort(key=lambda item: (item[1].grid_pos[0] + item[1].grid_pos[1], item[1].grid_pos[1]))

        for item_type, item in render_queue:
            if item_type == 'block':
                self._draw_iso_cube(item.grid_pos, 20, item.color)
                self._draw_health_bar(self._grid_to_screen(*item.grid_pos), item.health, item.max_health, 30)
            elif item_type == 'enemy':
                pos = (int(item.pixel_pos[0]), int(item.pixel_pos[1]))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1] - 5, 6, self.COLOR_ENEMY)
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1] - 5, 6, self.COLOR_ENEMY)
                self._draw_health_bar(pos, item.health, item.max_health, 25)
            elif item_type == 'fortress':
                self._draw_iso_cube(self.fortress_pos, 25, self.COLOR_FORTRESS)

        # Render cursor if in building phase
        if self.game_phase in ['BUILDING', 'BUILDING_START']:
            self._render_cursor()

        # Render projectiles and particles on top
        for p in self.projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p.pos[0]), int(p.pos[1]), 3, (100, 200, 255))
            pygame.gfxdraw.aacircle(self.screen, int(p.pos[0]), int(p.pos[1]), 3, (100, 200, 255))
        
        for p in self.particles:
            alpha = int(255 * (p.life / p.max_life))
            color = (*p.color, alpha)
            if p.size > 1:
                pygame.draw.circle(self.screen, color, (p.x, p.y), p.size)

    def _draw_iso_cube(self, grid_pos, height, color):
        x, y = self._grid_to_screen(*grid_pos)
        
        dark_color = tuple(max(0, c - 40) for c in color)
        light_color = tuple(min(255, c + 40) for c in color)

        # Top face
        points_top = [
            (x, y - height),
            (x + self.TILE_WIDTH_HALF, y - height + self.TILE_HEIGHT_HALF),
            (x, y - height + 2 * self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF, y - height + self.TILE_HEIGHT_HALF)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points_top, light_color)
        pygame.gfxdraw.aapolygon(self.screen, points_top, light_color)

        # Left face
        points_left = [
            (x - self.TILE_WIDTH_HALF, y - height + self.TILE_HEIGHT_HALF),
            (x, y - height + 2 * self.TILE_HEIGHT_HALF),
            (x, y + 2 * self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points_left, color)
        pygame.gfxdraw.aapolygon(self.screen, points_left, color)

        # Right face
        points_right = [
            (x, y - height + 2 * self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF, y - height + self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
            (x, y + 2 * self.TILE_HEIGHT_HALF)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points_right, dark_color)
        pygame.gfxdraw.aapolygon(self.screen, points_right, dark_color)

    def _render_cursor(self):
        x, y = self.cursor_pos
        is_valid = self.grid[x][y] is None and (x, y) != self.fortress_pos
        color = self.COLOR_CURSOR_VALID if is_valid else self.COLOR_CURSOR_INVALID
        
        sx, sy = self._grid_to_screen(x, y)
        points = [
            (sx, sy),
            (sx + self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF),
            (sx, sy + 2 * self.TILE_HEIGHT_HALF),
            (sx - self.TILE_WIDTH_HALF, sy + self.TILE_HEIGHT_HALF)
        ]
        pygame.draw.polygon(self.screen, color, points, 2)

    def _render_ui(self):
        # Fortress Health
        health_text = self.font_medium.render(f"Fortress Health: {int(self.fortress_health)} / {self.fortress_max_health}", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (10, 10))
        
        # Wave Number
        wave_text = self.font_medium.render(f"Wave: {self.current_wave} / 20", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 10))

        # Inventory
        for i, block_type in enumerate(self.BLOCK_TYPES):
            y_pos = 40 + i * 25
            color = self.BLOCK_PROPERTIES[block_type]['color']
            pygame.draw.rect(self.screen, color, (10, y_pos, 15, 15))
            count = self.block_inventory[block_type]
            text = self.font_small.render(f"{block_type}: {count}", True, self.COLOR_TEXT)
            self.screen.blit(text, (30, y_pos))
            if i == self.selected_block_idx:
                 pygame.draw.rect(self.screen, (255, 255, 255), (8, y_pos - 2, text.get_width() + 26, 20), 1)

        # Game Phase Info
        if self.game_phase == 'BUILDING_START':
            msg = self.font_large.render(f"WAVE {self.current_wave}", True, self.COLOR_TEXT)
            self.screen.blit(msg, (self.SCREEN_WIDTH/2 - msg.get_width()/2, self.SCREEN_HEIGHT/2 - msg.get_height()/2))
        elif self.game_phase == 'BUILDING':
            msg = self.font_medium.render(f"Build Time: {math.ceil(self.phase_timer / 30)}s", True, self.COLOR_TEXT)
            self.screen.blit(msg, (self.SCREEN_WIDTH/2 - msg.get_width()/2, 10))
        
        # Game Over/Win
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            msg_text = "YOU WIN!" if self.win else "GAME OVER"
            msg = self.font_large.render(msg_text, True, self.COLOR_HEALTH_GREEN if self.win else self.COLOR_HEALTH_RED)
            self.screen.blit(msg, (self.SCREEN_WIDTH/2 - msg.get_width()/2, self.SCREEN_HEIGHT/2 - msg.get_height()/2))

    def _draw_health_bar(self, pos, current, maximum, y_offset, width=30):
        if current >= maximum: return
        bar_x = pos[0] - width // 2
        bar_y = pos[1] - y_offset
        ratio = max(0, current / maximum)
        
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_RED, (bar_x, bar_y, width, 5))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_GREEN, (bar_x, bar_y, width * ratio, 5))
        pygame.draw.rect(self.screen, (255,255,255), (bar_x, bar_y, width, 5), 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "fortress_health": self.fortress_health,
            "enemies_remaining": len(self.enemies),
            "game_phase": self.game_phase
        }

    def close(self):
        pygame.quit()

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

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset(seed=42)
    done = False
    
    # --- Manual Play Controls ---
    # This block allows a human to play the game.
    # It maps keyboard presses to the MultiDiscrete action space.
    
    # Set auto_advance to False for manual play to feel responsive
    env.auto_advance = False 
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.user_guide)
    print("="*30 + "\n")
    
    # For human play, we need to render to a display window
    # Create display if it doesn't exist
    pygame.display.init()
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Isometric Fortress Defense")

    while not done:
        # Reset actions at the start of each frame
        action = np.array([0, 0, 0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        
        # Map keys to actions
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Blit the observation surface to the display
        display_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(display_surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Limit to 30 FPS

    print(f"Game Over. Final Info: {info}")
    env.close()