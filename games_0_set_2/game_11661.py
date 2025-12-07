import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T12:22:59.244563
# Source Brief: brief_01661.md
# Brief Index: 1661
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a Tetris-like game with three simultaneously falling blocks.
    The goal is to clear 10 lines by arranging the blocks. Gravity increases with each line clear.

    Action Space: MultiDiscrete([5, 2, 2])
    - actions[0]: Movement (0=none, 1=up(noop), 2=hard_drop, 3=left, 4=right)
    - actions[1]: Space button (0=released, 1=held) -> Used to cycle active block.
    - actions[2]: Shift button (0=released, 1=held) -> No-op.

    Observation Space: Box(0, 255, (400, 640, 3), uint8) - A rendered frame of the game.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A Tetris-like game where three blocks fall at once. "
        "Cycle between blocks and arrange them to clear lines."
    )
    user_guide = (
        "Controls: ←→ to move the active block, ↓ to hard drop. "
        "Press space to cycle which block is active."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_COLS, PLAYFIELD_ROWS = 10, 20
    BLOCK_SIZE = 18
    PLAYFIELD_WIDTH = PLAYFIELD_COLS * BLOCK_SIZE
    PLAYFIELD_HEIGHT = PLAYFIELD_ROWS * BLOCK_SIZE
    PLAYFIELD_X = (SCREEN_WIDTH - PLAYFIELD_WIDTH) // 2
    PLAYFIELD_Y = (SCREEN_HEIGHT - PLAYFIELD_HEIGHT) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (30, 30, 45)
    COLOR_BORDER = (80, 80, 100)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_SHADOW = (10, 10, 15)
    BLOCK_COLORS = [
        (230, 50, 50),   # Red
        (50, 230, 50),   # Green
        (50, 150, 230),  # Blue
        (230, 230, 50),  # Yellow
        (230, 50, 230),  # Magenta
        (50, 230, 230),  # Cyan
    ]
    SPECIAL_BLOCK_COLOR = (255, 165, 0) # Orange
    
    # Game parameters
    WIN_CONDITION_LINES = 10
    MAX_STEPS = 1500
    INITIAL_GRAVITY = 0.03 # blocks per step
    GRAVITY_INCREASE_PER_COMBO = 0.01
    SPECIAL_BLOCK_FREQUENCY = 20 # Every 20 blocks spawned

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
        try:
            self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
            self.font_small = pygame.font.SysFont("Consolas", 18)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 30)
            self.font_small = pygame.font.Font(None, 24)

        self.render_mode = render_mode
        # The following method is called in reset(), no need to call it here.
        # self._initialize_state()

    def _initialize_state(self):
        """Initializes all game state variables. Called by reset()."""
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        # Game grid: 0 for empty, otherwise color index + 1
        self.grid = np.zeros((self.PLAYFIELD_COLS, self.PLAYFIELD_ROWS + 4), dtype=int)
        
        self.gravity = self.INITIAL_GRAVITY
        self.active_block_index = 0
        self._prev_space_held = False
        self.block_spawn_counter = 0

        self.falling_blocks = []
        self.next_blocks = [self._create_random_block() for _ in range(3)]
        self._spawn_new_blocks()

        self.particles = []
        self.line_clear_flash = [] # (y_row, alpha)

    def _create_random_block(self):
        self.block_spawn_counter += 1
        is_special = (self.block_spawn_counter % self.SPECIAL_BLOCK_FREQUENCY == 0)
        return {
            "color_idx": len(self.BLOCK_COLORS) if is_special else self.np_random.integers(0, len(self.BLOCK_COLORS)),
            "is_special": is_special
        }

    def _spawn_new_blocks(self):
        initial_positions = [2, 4, 7]
        for i in range(3):
            block_info = self.next_blocks[i]
            self.falling_blocks.append({
                "x": initial_positions[i],
                "y": 0.0,
                "color_idx": block_info["color_idx"],
                "is_special": block_info["is_special"]
            })
        self.next_blocks = [self._create_random_block() for _ in range(3)]
        
        # Check for immediate game over
        for block in self.falling_blocks:
            if self.grid[block["x"], math.floor(block["y"])] != 0:
                self.game_over = True
                break

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._initialize_state()
        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        reward = 0
        
        if self.game_over:
            terminated = self._check_termination()
            reward += -100 if self.game_over else 0
            return self._get_observation(), reward, terminated, False, self._get_info()

        self._handle_input(action)
        reward += self._update_physics()
        
        if not self.falling_blocks:
            combo_reward = self._check_and_clear_lines()
            reward += combo_reward
            if not self.game_over:
                self._spawn_new_blocks()

        self._update_effects()

        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        
        if terminated and not truncated:
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                reward += 100 # Win bonus
            elif self.game_over:
                reward += -100 # Loss penalty
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        if not self.falling_blocks:
            return
        active_block = self.falling_blocks[self.active_block_index]

        # Cycle active block on space press
        if space_held and not self._prev_space_held:
            # SFX: select_beep
            self.active_block_index = (self.active_block_index + 1) % len(self.falling_blocks)
        self._prev_space_held = space_held

        # Horizontal movement
        if movement in [3, 4]: # 3=left, 4=right
            dx = -1 if movement == 3 else 1
            new_x = active_block['x'] + dx
            if 0 <= new_x < self.PLAYFIELD_COLS and self.grid[new_x, int(active_block['y'])] == 0:
                active_block['x'] = new_x
        
        # Hard drop
        if movement == 2:
            # SFX: hard_drop
            while not self._check_collision(active_block):
                active_block['y'] += 1
            active_block['y'] -= 1

    def _update_physics(self):
        landed_reward = 0
        landed_blocks = []
        for i, block in enumerate(self.falling_blocks):
            block['y'] += self.gravity
            if self._check_collision(block):
                block['y'] = math.floor(block['y'])
                self._lock_block(block)
                landed_blocks.append(i)
                landed_reward += 0.1 # Reward for placing a block
        
        # Remove landed blocks in reverse to avoid index errors
        for i in sorted(landed_blocks, reverse=True):
            if i < self.active_block_index:
                self.active_block_index -= 1
            del self.falling_blocks[i]
        
        if self.falling_blocks and self.active_block_index >= len(self.falling_blocks):
            self.active_block_index = 0 if self.falling_blocks else -1

        return landed_reward

    def _check_collision(self, block):
        y_pos = math.floor(block['y'])
        if y_pos >= self.PLAYFIELD_ROWS:
            return True
        if self.grid[block['x'], y_pos] != 0:
            return True
        return False

    def _lock_block(self, block):
        # SFX: block_land
        x, y = block['x'], int(block['y']) - 1
        if 0 <= y < self.PLAYFIELD_ROWS:
            self.grid[x, y] = block['color_idx'] + 1
            if block['is_special']:
                self.grid[x, y] *= -1 # Mark as special
        else: # Block landed above the visible playfield
            self.game_over = True
        
        # Create landing particles
        for _ in range(5):
            self.particles.append(self._create_particle(self.PLAYFIELD_X + x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2, 
                                                        self.PLAYFIELD_Y + y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2,
                                                        self._get_color_from_idx(block['color_idx'])))

    def _check_and_clear_lines(self):
        reward = 0
        lines_to_clear = []
        for y in range(self.PLAYFIELD_ROWS):
            if np.all(self.grid[:, y] != 0):
                lines_to_clear.append(y)

        if not lines_to_clear:
            return 0
        
        # SFX: line_clear
        reward += len(lines_to_clear)
        self.gravity += self.GRAVITY_INCREASE_PER_COMBO * len(lines_to_clear)
        
        cleared_mask = np.zeros_like(self.grid, dtype=bool)
        
        # Handle special blocks first
        for y in lines_to_clear:
            self.line_clear_flash.append([y, 255])
            for x in range(self.PLAYFIELD_COLS):
                if self.grid[x, y] < 0: # Is a special block
                    reward += 5
                    # SFX: special_ability
                    # Mark cross shape for clearing
                    for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.PLAYFIELD_COLS and 0 <= ny < self.PLAYFIELD_ROWS:
                            cleared_mask[nx, ny] = True
        
        # Mark full lines for clearing
        for y in lines_to_clear:
            cleared_mask[:, y] = True

        # Drop columns down
        new_grid = np.zeros_like(self.grid)
        for x in range(self.PLAYFIELD_COLS):
            dest_y = self.PLAYFIELD_ROWS - 1
            for y in range(self.PLAYFIELD_ROWS - 1, -1, -1):
                if not cleared_mask[x, y]:
                    new_grid[x, dest_y] = self.grid[x, y]
                    dest_y -= 1
        
        self.lines_cleared += len(lines_to_clear)
        self.grid = new_grid
        return reward

    def _update_effects(self):
        # Update particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # particle gravity
            p['life'] -= 1
        
        # Update line clear flash
        self.line_clear_flash = [f for f in self.line_clear_flash if f[1] > 0]
        for f in self.line_clear_flash:
            f[1] -= 20 # fade out speed

    def _check_termination(self):
        return (self.game_over or 
                self.lines_cleared >= self.WIN_CONDITION_LINES)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "gravity": self.gravity
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_playfield()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_playfield(self):
        # Draw border and grid lines
        pygame.draw.rect(self.screen, self.COLOR_BORDER, 
                         (self.PLAYFIELD_X - 4, self.PLAYFIELD_Y - 4, 
                          self.PLAYFIELD_WIDTH + 8, self.PLAYFIELD_HEIGHT + 8), 4, border_radius=5)
        for x in range(1, self.PLAYFIELD_COLS):
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.PLAYFIELD_X + x * self.BLOCK_SIZE, self.PLAYFIELD_Y),
                             (self.PLAYFIELD_X + x * self.BLOCK_SIZE, self.PLAYFIELD_Y + self.PLAYFIELD_HEIGHT))
        for y in range(1, self.PLAYFIELD_ROWS):
            pygame.draw.line(self.screen, self.COLOR_GRID,
                             (self.PLAYFIELD_X, self.PLAYFIELD_Y + y * self.BLOCK_SIZE),
                             (self.PLAYFIELD_X + self.PLAYFIELD_WIDTH, self.PLAYFIELD_Y + y * self.BLOCK_SIZE))

        # Draw locked blocks
        for x in range(self.PLAYFIELD_COLS):
            for y in range(self.PLAYFIELD_ROWS):
                if self.grid[x, y] != 0:
                    self._draw_block(self.screen, x, y, self.grid[x, y])

        # Draw falling blocks
        for i, block in enumerate(self.falling_blocks):
            is_active = (i == self.active_block_index)
            self._draw_block(self.screen, block['x'], block['y'], block['color_idx'] + 1, is_falling=True, is_active=is_active)

        # Draw effects
        self._render_effects()

    def _draw_block(self, surface, x, y, color_val, is_falling=False, is_active=False):
        is_special = color_val < 0
        color_idx = abs(color_val) - 1
        color = self._get_color_from_idx(color_idx)
        
        px = self.PLAYFIELD_X + x * self.BLOCK_SIZE
        py = self.PLAYFIELD_Y + (y if is_falling else y) * self.BLOCK_SIZE

        if is_active:
            glow_rect = pygame.Rect(px - 3, py - 3, self.BLOCK_SIZE + 6, self.BLOCK_SIZE + 6)
            pygame.draw.rect(surface, (255, 255, 255), glow_rect, 0, border_radius=5)

        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        # Bevel effect
        light_color = tuple(min(255, c + 40) for c in color)
        dark_color = tuple(max(0, c - 40) for c in color)
        pygame.draw.rect(surface, dark_color, rect, 0, border_radius=3)
        pygame.draw.rect(surface, color, rect.inflate(-4, -4), 0, border_radius=2)
        
        # Top-left highlight
        pygame.draw.line(surface, light_color, (px + 2, py + 2), (px + self.BLOCK_SIZE - 3, py + 2), 1)
        pygame.draw.line(surface, light_color, (px + 2, py + 2), (px + 2, py + self.BLOCK_SIZE - 3), 1)

        if is_special:
            star_center_x, star_center_y = px + self.BLOCK_SIZE // 2, py + self.BLOCK_SIZE // 2
            points = []
            for i in range(5):
                angle = math.radians(i * 72 * 2 + 90) # 5 points, offset by 90 deg
                outer_r = self.BLOCK_SIZE * 0.4
                inner_r = self.BLOCK_SIZE * 0.15
                points.append((star_center_x + math.cos(angle) * outer_r, star_center_y - math.sin(angle) * outer_r))
                angle += math.radians(36)
                points.append((star_center_x + math.cos(angle) * inner_r, star_center_y - math.sin(angle) * inner_r))
            pygame.gfxdraw.aapolygon(surface, points, (255, 255, 255))
            pygame.gfxdraw.filled_polygon(surface, points, (255, 255, 255))


    def _get_color_from_idx(self, idx):
        return self.SPECIAL_BLOCK_COLOR if idx >= len(self.BLOCK_COLORS) else self.BLOCK_COLORS[idx]

    def _render_effects(self):
        # Particles
        for p in self.particles:
            alpha = max(0, min(255, p['life'] * 10))
            color = p['color'] + (alpha,)
            temp_surf = pygame.Surface((3, 3), pygame.SRCALPHA)
            pygame.draw.rect(temp_surf, color, (0, 0, 3, 3))
            self.screen.blit(temp_surf, (int(p['x']), int(p['y'])))

        # Line clear flash
        for y, alpha in self.line_clear_flash:
            flash_surface = pygame.Surface((self.PLAYFIELD_WIDTH, self.BLOCK_SIZE), pygame.SRCALPHA)
            flash_surface.fill((255, 255, 255, int(alpha)))
            self.screen.blit(flash_surface, (self.PLAYFIELD_X, self.PLAYFIELD_Y + y * self.BLOCK_SIZE))

    def _render_ui(self):
        # --- Helper to draw text with shadow ---
        def draw_text(text, font, color, x, y, shadow_color=self.COLOR_TEXT_SHADOW):
            shadow_surf = font.render(text, True, shadow_color)
            self.screen.blit(shadow_surf, (x + 2, y + 2))
            text_surf = font.render(text, True, color)
            self.screen.blit(text_surf, (x, y))

        # --- Left Panel: Stats ---
        left_panel_x = 30
        draw_text(f"LINES", self.font_small, self.COLOR_BORDER, left_panel_x, 50)
        draw_text(f"{self.lines_cleared} / {self.WIN_CONDITION_LINES}", self.font_large, self.COLOR_TEXT, left_panel_x, 70)
        
        draw_text(f"GRAVITY", self.font_small, self.COLOR_BORDER, left_panel_x, 130)
        draw_text(f"{self.gravity * 100 / self.INITIAL_GRAVITY:.0f}%", self.font_large, self.COLOR_TEXT, left_panel_x, 150)
        
        draw_text(f"STEPS", self.font_small, self.COLOR_BORDER, left_panel_x, 210)
        draw_text(f"{self.steps}", self.font_large, self.COLOR_TEXT, left_panel_x, 230)

        # --- Right Panel: Next Blocks ---
        right_panel_x = self.SCREEN_WIDTH - 150
        draw_text("NEXT", self.font_large, self.COLOR_TEXT, right_panel_x, 50)
        
        preview_bg_rect = (right_panel_x - 10, 85, 100, 200)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_bg_rect, 0, border_radius=5)
        pygame.draw.rect(self.screen, self.COLOR_BORDER, preview_bg_rect, 2, border_radius=5)

        for i, block_info in enumerate(self.next_blocks):
            y_offset = i * (self.BLOCK_SIZE * 2.5)
            # Create a temporary surface for each block to respect panel boundaries
            block_surf = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            self._draw_block(block_surf, 0, 0, block_info['color_idx'] + 1 if not block_info['is_special'] else -(len(self.BLOCK_COLORS)+1))
            self.screen.blit(block_surf, (right_panel_x + 30, 110 + y_offset))

    def _create_particle(self, x, y, color):
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(1, 3)
        return {
            'x': x, 'y': y,
            'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
            'life': self.np_random.integers(20, 40),
            'color': color
        }

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    # --- Example Usage ---
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This part allows a human to play the game for testing purposes.
    # It will not be used by the Gymnasium runner.
    
    # Remap keyboard keys to the MultiDiscrete action space
    key_to_action = {
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_DOWN:  [2, 0, 0], # Hard Drop
        pygame.K_SPACE: [0, 1, 0], # Cycle Block
    }
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    
    # Setup a display window for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS", etc.
    pygame.display.init()
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("TrioFall")
    clock = pygame.time.Clock()
    
    running = True
    while running and not terminated and not truncated:
        action = [0, 0, 0] # Default action: no-op, buttons released
        
        # Handle held keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Handle single-press events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    action[0] = 2 # Hard drop
                if event.key == pygame.K_SPACE:
                    action[1] = 1 # Press space
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation from the environment to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for playability
        
        if terminated or truncated:
            print(f"Game Over! Final Lines: {info['lines_cleared']}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()