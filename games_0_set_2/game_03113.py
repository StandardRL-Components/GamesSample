
# Generated: 2025-08-28T07:01:31.329940
# Source Brief: brief_03113.md
# Brief Index: 3113

        
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
        "Controls: ←→ to move the block, ↓ to drop it one cell. The block is placed when it cannot drop further."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Place falling blocks to clear lines from the grid. The goal is to clear 80% of the board without getting stuck."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.CELL_SIZE = 32
        self.GRID_PIXEL_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.GRID_PIXEL_HEIGHT = self.GRID_HEIGHT * self.CELL_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_PIXEL_WIDTH) // 2
        self.GRID_OFFSET_Y = self.SCREEN_HEIGHT - self.GRID_PIXEL_HEIGHT - 20 # 20px padding at bottom

        self.MAX_STEPS = 1000
        self.WIN_CLEARANCE = 0.8

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 220, 240)
        self.BLOCK_COLORS = [
            (255, 87, 87),    # Red
            (87, 255, 87),    # Green
            (87, 87, 255),    # Blue
            (255, 255, 87),   # Yellow
            (87, 255, 255),   # Cyan
            (255, 87, 255),   # Magenta
        ]

        # --- Block Shapes (pivot at (0,0)) ---
        self.SHAPES = [
            [(0, 0), (0, -1), (0, 1), (0, 2)],    # I shape
            [(0, 0), (1, 0), (0, 1), (1, 1)],    # O shape
            [(0, 0), (-1, 0), (1, 0), (0, -1)],   # T shape
            [(0, 0), (0, -1), (0, 1), (1, 1)],    # L shape
            [(0, 0), (0, -1), (0, 1), (-1, 1)],   # J shape
            [(0, 0), (-1, 0), (0, -1), (1, -1)],  # S shape
            [(0, 0), (1, 0), (0, -1), (-1, -1)],  # Z shape
        ]

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # --- State Variables ---
        self.grid = None
        self.grid_age = None
        self.current_block_shape = None
        self.current_block_pos = None
        self.current_block_color_idx = None
        self.steps = 0
        self.score = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.grid_age = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._spawn_block()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        
        placed_block = False

        # --- Action Handling ---
        if movement in [2, 3, 4]: # Down, Left, Right
            dx, dy = 0, 0
            if movement == 2: dy = 1   # Down
            if movement == 3: dx = -1  # Left
            if movement == 4: dx = 1   # Right

            new_pos = (self.current_block_pos[0] + dx, self.current_block_pos[1] + dy)
            
            if not self._check_collision(self.current_block_shape, new_pos):
                self.current_block_pos = new_pos
            elif movement == 2: # Collision while trying to move down means placement
                self._place_block()
                reward += 0.1 # Reward for placing a block
                placed_block = True

        # --- Post-Placement Logic ---
        if placed_block:
            lines_cleared = self._clear_lines()
            if lines_cleared > 0:
                reward += lines_cleared * 1.0
                self.score += lines_cleared
                # sound: line_clear.wav
            
            self._spawn_block() # This can set self.game_over

        self.steps += 1
        
        # --- Termination Check ---
        terminated = self._check_termination()
        if terminated:
            if self._calculate_clearance() >= self.WIN_CLEARANCE:
                reward = 100 # Win
                # sound: win.wav
            elif self.game_over: # Loss by no moves
                reward = -100 # Loss
                # sound: game_over.wav
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_block(self):
        shape_idx = self.np_random.integers(0, len(self.SHAPES))
        self.current_block_shape = self.SHAPES[shape_idx]
        self.current_block_color_idx = self.np_random.integers(0, len(self.BLOCK_COLORS))
        
        spawn_x = self.GRID_WIDTH // 2
        spawn_y = 0 
        
        # Adjust spawn_y if block starts above the grid
        min_y = min(p[1] for p in self.current_block_shape)
        spawn_y -= min_y
        
        self.current_block_pos = (spawn_x, spawn_y)

        if self._check_collision(self.current_block_shape, self.current_block_pos):
            self.game_over = True
            self.current_block_shape = None # No block to render

    def _place_block(self):
        if self.current_block_shape is None: return

        # Increment age of all existing blocks
        self.grid_age[self.grid > 0] += 1

        for dx, dy in self.current_block_shape:
            x = self.current_block_pos[0] + dx
            y = self.current_block_pos[1] + dy
            if 0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT:
                self.grid[y, x] = self.current_block_color_idx + 1 # Use 1-based index for color
                self.grid_age[y, x] = 0 # Reset age for new cells
        # sound: place_block.wav

    def _check_collision(self, shape, pos):
        for dx, dy in shape:
            x = pos[0] + dx
            y = pos[1] + dy
            
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True # Wall collision
            if self.grid[y, x] != 0:
                return True # Other block collision
        return False

    def _clear_lines(self):
        lines_to_clear = [r for r in range(self.GRID_HEIGHT) if np.all(self.grid[r, :] != 0)]
        if not lines_to_clear:
            return 0
        
        # Remove lines from bottom up
        for r in sorted(lines_to_clear, reverse=True):
            self.grid[1:r+1, :] = self.grid[0:r, :]
            self.grid[0, :] = 0
            self.grid_age[1:r+1, :] = self.grid_age[0:r, :]
            self.grid_age[0, :] = 0

        return len(lines_to_clear)

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        if self._calculate_clearance() >= self.WIN_CLEARANCE:
            return True
        return False

    def _calculate_clearance(self):
        filled_cells = np.count_nonzero(self.grid)
        return (self.GRID_WIDTH * self.GRID_HEIGHT - filled_cells) / (self.GRID_WIDTH * self.GRID_HEIGHT)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "clearance": self._calculate_clearance(),
        }
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_placed_blocks()
        self._render_falling_block()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + x * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_PIXEL_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_PIXEL_WIDTH, self.GRID_OFFSET_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos)

    def _render_placed_blocks(self):
        max_age = 40
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    age = self.grid_age[r, c]
                    # Recently filled (low age) are darker
                    gray_val = int(50 + min(age, max_age) * (150 / max_age))
                    color = (gray_val, gray_val, gray_val)
                    
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + c * self.CELL_SIZE,
                        self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_falling_block(self):
        if self.current_block_shape is None: return
        
        color = self.BLOCK_COLORS[self.current_block_color_idx]
        
        for dx, dy in self.current_block_shape:
            x = self.current_block_pos[0] + dx
            y = self.current_block_pos[1] + dy
            
            rect = pygame.Rect(
                self.GRID_OFFSET_X + x * self.CELL_SIZE,
                self.GRID_OFFSET_Y + y * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            # Draw a slightly smaller inner rect for a border effect
            inner_rect = rect.inflate(-4, -4)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, tuple(min(255, c + 50) for c in color), inner_rect)

    def _render_ui(self):
        clearance_pct = self._calculate_clearance() * 100
        clearance_text = f"CLEARANCE: {clearance_pct:.1f}%"
        
        text_surf = self.font_large.render(clearance_text, True, self.COLOR_TEXT)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, 40))
        self.screen.blit(text_surf, text_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win = self._calculate_clearance() >= self.WIN_CLEARANCE
            end_text_str = "GOAL REACHED!" if win else "GAME OVER"
            end_text_surf = self.font_large.render(end_text_str, True, (255, 255, 100))
            end_text_rect = end_text_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text_surf, end_text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Grid Clearer")
    clock = pygame.time.Clock()
    
    print(env.user_guide)
    print(env.game_description)

    while not terminated:
        movement = 0 # No-op by default
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_UP:
                    # movement = 1 # Not implemented
                    pass
                elif event.key == pygame.K_r:
                    obs, info = env.reset() # Reset on 'r' key
        
        if movement != 0:
            action = [movement, 0, 0] # space/shift not used
            obs, reward, term, trunc, info = env.step(action)
            terminated = term
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}, Clearance: {info['clearance']:.2f}")

        # Draw the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print("Game Over!")
            pygame.time.wait(3000) # Wait 3 seconds before closing
            
        clock.tick(30) # Limit frame rate
        
    env.close()