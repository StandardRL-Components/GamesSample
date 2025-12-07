
# Generated: 2025-08-27T13:40:12.658661
# Source Brief: brief_00442.md
# Brief Index: 442

        
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


# Helper classes for game objects
class Block:
    def __init__(self, grid_x, grid_y, block_type_info):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.type_info = block_type_info
        self.max_health = block_type_info['max_health']
        self.health = self.max_health
        self.color = block_type_info['color']
        self.hit_timer = 0

    def take_damage(self, amount):
        self.health -= amount
        self.hit_timer = 5  # Flash for 5 frames
        return self.health <= 0

class Enemy:
    def __init__(self, wave_num, grid_size, cell_size, grid_offset_x, grid_offset_y, np_random):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_offset_x = grid_offset_x
        self.grid_offset_y = grid_offset_y

        # Spawn on a random edge
        edge = np_random.integers(0, 4)
        if edge == 0:  # Top
            self.x = np_random.uniform(self.grid_offset_x, self.grid_offset_x + self.grid_size * self.cell_size)
            self.y = self.grid_offset_y - self.cell_size
        elif edge == 1:  # Bottom
            self.x = np_random.uniform(self.grid_offset_x, self.grid_offset_x + self.grid_size * self.cell_size)
            self.y = self.grid_offset_y + (self.grid_size + 1) * self.cell_size
        elif edge == 2:  # Left
            self.x = self.grid_offset_x - self.cell_size
            self.y = np_random.uniform(self.grid_offset_y, self.grid_offset_y + self.grid_size * self.cell_size)
        else:  # Right
            self.x = self.grid_offset_x + (self.grid_size + 1) * self.cell_size
            self.y = np_random.uniform(self.grid_offset_y, self.grid_offset_y + self.grid_size * self.cell_size)

        self.max_health = 20 * (1 + wave_num * 0.2)
        self.health = self.max_health
        self.speed = 1.5 * (1 + wave_num * 0.1)
        self.damage = 10
        self.radius = 8
        self.color = (255, 50, 50)

    def move(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed

    def get_grid_pos(self):
        grid_x = int((self.x - self.grid_offset_x) / self.cell_size)
        grid_y = int((self.y - self.grid_offset_y) / self.cell_size)
        return grid_x, grid_y

class Particle:
    def __init__(self, x, y, color, life, size, velocity_spread, np_random):
        self.x = x
        self.y = y
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        angle = np_random.uniform(0, 2 * math.pi)
        speed = np_random.uniform(0.5, velocity_spread)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life <= 0

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / self.max_life))
            radius = int(self.size * (self.life / self.max_life))
            if radius > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*self.color, alpha), (radius, radius), radius)
                surface.blit(temp_surf, (int(self.x - radius), int(self.y - radius)), special_flags=pygame.BLEND_RGBA_ADD)


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to place a block. "
        "Hold Shift to cycle through block types (Wood, Stone, Steel)."
    )

    game_description = (
        "Build a block fortress to defend against waves of enemies. Survive 10 waves to win. "
        "If your fortress health reaches zero, you lose."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Spaces
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.W, self.H = 640, 400
        self.screen = pygame.Surface((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_FORTRESS = (30, 30, 50)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_HEALTH_BAR = (40, 200, 80)
        self.COLOR_HEALTH_BAR_BG = (80, 40, 40)
        self.COLOR_HIT_FLASH = (255, 255, 255)

        # Game constants
        self.MAX_STEPS = 2000
        self.MAX_WAVES = 10
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.GRID_OFFSET_X = (self.W - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_OFFSET_Y = (self.H - self.GRID_SIZE * self.CELL_SIZE) // 2 + 20
        self.FORTRESS_ZONE = pygame.Rect(
            self.GRID_OFFSET_X + 3 * self.CELL_SIZE,
            self.GRID_OFFSET_Y + 3 * self.CELL_SIZE,
            4 * self.CELL_SIZE,
            4 * self.CELL_SIZE
        )
        self.FORTRESS_CENTER_X = self.FORTRESS_ZONE.centerx
        self.FORTRESS_CENTER_Y = self.FORTRESS_ZONE.centery
        
        # Block types
        self.block_types = [
            {'name': 'Wood', 'max_health': 100, 'color': (160, 82, 45)},
            {'name': 'Stone', 'max_health': 300, 'color': (112, 128, 144)},
            {'name': 'Steel', 'max_health': 800, 'color': (180, 190, 200)},
        ]
        
        # Initialize state variables
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.fortress_health = 0
        self.max_fortress_health = 100
        self.current_wave = 0
        self.grid = {}
        self.cursor_pos = [0, 0]
        self.last_space_held = False
        self.last_shift_held = False
        self.selected_block_type_idx = 0
        self.enemies = []
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.fortress_health = self.max_fortress_health
        self.current_wave = 0
        self.grid.clear()
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.last_space_held = False
        self.last_shift_held = False
        self.selected_block_type_idx = 0
        self.enemies.clear()
        self.particles.clear()
        
        self._start_next_wave()
        
        return self._get_observation(), self._get_info()
    
    def _start_next_wave(self):
        self.current_wave += 1
        if self.current_wave > self.MAX_WAVES:
            self.win = True
            self.game_over = True
            return

        num_enemies = 2 + self.current_wave
        for _ in range(num_enemies):
            enemy = Enemy(self.current_wave, self.GRID_SIZE, self.CELL_SIZE, self.GRID_OFFSET_X, self.GRID_OFFSET_Y, self.np_random)
            self.enemies.append(enemy)

    def step(self, action):
        reward = 0.0
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1

        # --- 1. Process Player Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)

        if shift_pressed:
            self.selected_block_type_idx = (self.selected_block_type_idx + 1) % len(self.block_types)

        if space_pressed:
            pos_tuple = tuple(self.cursor_pos)
            if pos_tuple not in self.grid:
                block_info = self.block_types[self.selected_block_type_idx]
                self.grid[pos_tuple] = Block(self.cursor_pos[0], self.cursor_pos[1], block_info)
                # Small penalty for using resources
                reward -= 0.001

        # --- 2. Update Game World ---
        # Update block hit timers
        for block in self.grid.values():
            if block.hit_timer > 0:
                block.hit_timer -= 1

        # Update enemies
        destroyed_blocks = []
        for enemy in reversed(self.enemies):
            enemy.move(self.FORTRESS_CENTER_X, self.FORTRESS_CENTER_Y)
            
            grid_x, grid_y = enemy.get_grid_pos()
            
            attacked = False
            if 0 <= grid_x < self.GRID_SIZE and 0 <= grid_y < self.GRID_SIZE:
                pos_tuple = (grid_x, grid_y)
                if pos_tuple in self.grid:
                    attacked = True
                    block = self.grid[pos_tuple]
                    # sfx: block_hit
                    if block.take_damage(enemy.damage):
                        destroyed_blocks.append(pos_tuple)
                        reward -= 0.01
                    
                    for _ in range(3): self.particles.append(Particle(enemy.x, enemy.y, block.color, 15, 4, 1.5, self.np_random))
            
            if not attacked:
                enemy_rect = pygame.Rect(enemy.x - enemy.radius, enemy.y - enemy.radius, enemy.radius * 2, enemy.radius * 2)
                if self.FORTRESS_ZONE.colliderect(enemy_rect):
                    # sfx: fortress_hit
                    self.fortress_health -= enemy.damage
                    self.enemies.remove(enemy)
                    for _ in range(10): self.particles.append(Particle(enemy.x, enemy.y, self.COLOR_HEALTH_BAR, 20, 5, 2, self.np_random))

        for pos in destroyed_blocks:
            if pos in self.grid:
                # sfx: block_destroy
                block_center_x = self.GRID_OFFSET_X + pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
                block_center_y = self.GRID_OFFSET_Y + pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                for _ in range(20): self.particles.append(Particle(block_center_x, block_center_y, self.grid[pos].color, 30, 6, 3, self.np_random))
                del self.grid[pos]

        # Update particles
        self.particles = [p for p in self.particles if not p.update()]

        # --- 3. Check for Wave Completion ---
        if not self.enemies and not self.win:
            reward += 1.0  # Wave survival bonus
            # sfx: wave_complete
            self._start_next_wave()
            if self.win:
                reward += 100.0  # Win bonus
                # sfx: game_win

        # --- 4. Check Termination Conditions ---
        if self.fortress_health <= 0 and not self.game_over:
            self.fortress_health = 0
            self.game_over = True
            reward -= 100.0  # Loss penalty
            # sfx: game_over
            
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        if self.steps >= self.MAX_STEPS and not self.game_over:
             reward -= 50.0 # Time out penalty

        self.score += reward
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw fortress zone
        pygame.draw.rect(self.screen, self.COLOR_FORTRESS, self.FORTRESS_ZONE)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE))
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, y))

        # Draw blocks
        for pos, block in self.grid.items():
            rect = pygame.Rect(self.GRID_OFFSET_X + pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            
            health_ratio = max(0, block.health / block.max_health)
            color = tuple(int(c * (0.6 + 0.4 * health_ratio)) for c in block.color)
            
            pygame.draw.rect(self.screen, color, rect.inflate(-4, -4))
            pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in color), rect.inflate(-4,-4), 2)

            if block.hit_timer > 0:
                flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                alpha = int(200 * (block.hit_timer / 5))
                pygame.draw.rect(flash_surface, (*self.COLOR_HIT_FLASH, alpha), (0, 0, self.CELL_SIZE, self.CELL_SIZE))
                self.screen.blit(flash_surface, rect.topleft)

        # Draw cursor
        cursor_rect = pygame.Rect(self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE, self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)
        cursor_fill = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        cursor_fill.fill((*self.COLOR_CURSOR, 30))
        self.screen.blit(cursor_fill, cursor_rect.topleft)

        # Draw particles
        for p in self.particles:
            p.draw(self.screen)

        # Draw enemies
        for enemy in self.enemies:
            pos = (int(enemy.x), int(enemy.y))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy.radius, (0,0,0))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy.radius, (0,0,0))
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], enemy.radius-1, enemy.color)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], enemy.radius-1, enemy.color)

    def _render_ui(self):
        # Draw wave counter
        wave_text = self.font_medium.render(f"Wave: {self.current_wave} / {self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (20, 15))

        # Draw score
        score_text = self.font_medium.render(f"Score: {self.score:.2f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.W - score_text.get_width() - 20, 15))

        # Draw fortress health bar
        bar_width = self.W - 40
        bar_height = 20
        health_ratio = self.fortress_health / self.max_fortress_health
        
        bg_rect = pygame.Rect(20, self.H - bar_height - 15, bar_width, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, bg_rect, border_radius=5)
        
        fill_rect = pygame.Rect(20, self.H - bar_height - 15, bar_width * health_ratio, bar_height)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, fill_rect, border_radius=5)

        health_text = self.font_small.render("FORTRESS HEALTH", True, self.COLOR_TEXT)
        self.screen.blit(health_text, (bg_rect.centerx - health_text.get_width() // 2, bg_rect.y - 20))

        # Draw selected block
        block_type = self.block_types[self.selected_block_type_idx]
        block_text = self.font_medium.render(f"Selected: {block_type['name']}", True, self.COLOR_TEXT)
        self.screen.blit(block_text, (self.W // 2 - block_text.get_width() // 2, 15))
        pygame.draw.rect(self.screen, block_type['color'], (self.W // 2 + block_text.get_width() // 2 + 10, 18, 25, 25), border_radius=3)
        pygame.draw.rect(self.screen, tuple(int(c*0.7) for c in block_type['color']), (self.W // 2 + block_text.get_width() // 2 + 10, 18, 25, 25), 2, border_radius=3)


        # Draw game over screen
        if self.game_over:
            overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                end_text = self.font_large.render("VICTORY", True, (100, 255, 100))
            else:
                end_text = self.font_large.render("GAME OVER", True, (255, 100, 100))
            
            self.screen.blit(end_text, (self.W // 2 - end_text.get_width() // 2, self.H // 2 - end_text.get_height() // 2))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "fortress_health": self.fortress_health,
            "blocks": len(self.grid),
            "enemies": len(self.enemies),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        
        # Test game-specific assertions
        self.reset()
        assert self.fortress_health == self.max_fortress_health
        assert 0 <= self.current_wave <= self.MAX_WAVES + 1
        assert len(self.enemies) == 2 + self.current_wave

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Game loop
    running = True
    while running:
        # Pygame event handling
        action = np.array([0, 0, 0]) # no-op, released, released
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get keyboard state
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # For a real display, you'd create a window
        # For this example, we'll just confirm it works
        # If you want to see it, uncomment the following:
        
        # if 'display' not in locals():
        #     pygame.display.init()
        #     display = pygame.display.set_mode((640, 400))
        #     pygame.display.set_caption("Block Fortress")
        
        # display.blit(surf, (0,0))
        # pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Wave: {info['wave']}")
            obs, info = env.reset()

        # Since auto_advance is False, we need a delay to make it playable
        pygame.time.wait(50) 

    env.close()