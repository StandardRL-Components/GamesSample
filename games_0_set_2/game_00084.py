
# Generated: 2025-08-27T12:33:47.420852
# Source Brief: brief_00084.md
# Brief Index: 84

        
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
        "Controls: Use arrow keys to slide all blocks Up, Down, Left, or Right. "
        "Match 3 or more to clear them."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Slide colored blocks on a grid to match and eliminate them. "
        "Clear the board before you run out of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Game Constants ---
    GRID_DIM = 8
    NUM_COLORS = 5
    TOTAL_MOVES = 20
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # --- Rewards ---
    REWARD_MATCH_PER_BLOCK = 1
    REWARD_NO_MATCH_MOVE = -0.2
    REWARD_ROW_COL_CLEAR = 5
    REWARD_WIN = 100
    REWARD_LOSE = -50

    # --- Visuals ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 50, 60)
    BLOCK_COLORS = [
        (0, 0, 0),  # 0: Empty
        (255, 80, 80),   # 1: Red
        (80, 255, 80),   # 2: Green
        (80, 150, 255),  # 3: Blue
        (255, 255, 80),  # 4: Yellow
        (200, 80, 255),  # 5: Purple
    ]
    BLOCK_SHADOW_COLORS = [
        (0, 0, 0),
        (180, 50, 50),
        (50, 180, 50),
        (50, 100, 180),
        (180, 180, 50),
        (140, 50, 180),
    ]
    COLOR_FLASH = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 32)
        
        # Grid rendering properties
        self.grid_pixel_size = 360
        self.block_size = self.grid_pixel_size // self.GRID_DIM
        self.grid_start_x = (self.SCREEN_WIDTH - self.grid_pixel_size) // 2
        self.grid_start_y = (self.SCREEN_HEIGHT - self.grid_pixel_size) // 2
        
        # Initialize state variables
        self.grid = None
        self.moves_remaining = 0
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.blocks_to_flash = set()
        self.last_slide_direction = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.moves_remaining = self.TOTAL_MOVES
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.blocks_to_flash = set()
        self.last_slide_direction = 0

        self._generate_board()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Update game logic
        self.steps += 1
        self.blocks_to_flash.clear()
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        self.last_slide_direction = movement
        
        reward = 0
        terminated = False

        if movement == 0: # No-op action
            return self._get_observation(), reward, terminated, False, self._get_info()

        self.moves_remaining -= 1
        
        original_grid = self.grid.copy()
        self._slide_blocks(movement)
        grid_changed = not np.array_equal(self.grid, original_grid)

        total_blocks_removed = 0
        
        if grid_changed:
            # Cascade loop: find matches, remove, apply gravity, repeat
            while True:
                matches = self._find_all_matches(self.grid)
                if not matches:
                    break

                # Store pre-gravity state to check for line clears
                grid_before_gravity = self.grid.copy()
                for r, c in matches:
                    grid_before_gravity[r, c] = 0
                
                # Check for row/column clear bonuses
                reward += self._check_and_apply_clear_bonus(grid_before_gravity)

                num_removed = len(matches)
                total_blocks_removed += num_removed
                self.blocks_to_flash.update(matches)
                # Sfx: block_match_sound()

                for r, c in matches:
                    self.grid[r, c] = 0
                
                self._apply_gravity()
                self._refill_board()

        if total_blocks_removed > 0:
            reward += total_blocks_removed * self.REWARD_MATCH_PER_BLOCK
            self.score += total_blocks_removed * 10
        else:
            reward += self.REWARD_NO_MATCH_MOVE
            # Sfx: no_match_sound()
        
        terminated = self._check_termination()
        if terminated:
            if np.all(self.grid == 0): # Win condition
                reward += self.REWARD_WIN
            else: # Loss condition
                reward += self.REWARD_LOSE
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _check_termination(self):
        if self.game_over:
            return True
        if np.all(self.grid == 0): # Board cleared
            self.game_over = True
            return True
        if self.moves_remaining <= 0:
            self.game_over = True
            return True
        return False
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_remaining": self.moves_remaining,
        }

    def _slide_blocks(self, direction):
        # 1=up, 2=down, 3=left, 4=right
        if direction in [3, 4]: # Left/Right
            for r in range(self.GRID_DIM):
                row = self.grid[r, :]
                non_zeros = row[row != 0]
                new_row = np.zeros_like(row)
                if direction == 3: # Left
                    new_row[:len(non_zeros)] = non_zeros
                else: # Right
                    new_row[self.GRID_DIM - len(non_zeros):] = non_zeros
                self.grid[r, :] = new_row
        elif direction in [1, 2]: # Up/Down
            for c in range(self.GRID_DIM):
                col = self.grid[:, c]
                non_zeros = col[col != 0]
                new_col = np.zeros_like(col)
                if direction == 1: # Up
                    new_col[:len(non_zeros)] = non_zeros
                else: # Down
                    new_col[self.GRID_DIM - len(non_zeros):] = non_zeros
                self.grid[:, c] = new_col
        # Sfx: slide_sound()

    def _find_all_matches(self, grid):
        matches = set()
        # Horizontal matches
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM - 2):
                if grid[r, c] == 0: continue
                if grid[r, c] == grid[r, c+1] and grid[r, c] == grid[r, c+2]:
                    val = grid[r, c]
                    k = c + 3
                    while k < self.GRID_DIM and grid[r, k] == val:
                        k += 1
                    for i in range(c, k):
                        matches.add((r, i))
        
        # Vertical matches
        for c in range(self.GRID_DIM):
            for r in range(self.GRID_DIM - 2):
                if grid[r, c] == 0: continue
                if grid[r, c] == grid[r+1, c] and grid[r, c] == grid[r+2, c]:
                    val = grid[r, c]
                    k = r + 3
                    while k < self.GRID_DIM and grid[k, c] == val:
                        k += 1
                    for i in range(r, k):
                        matches.add((i, c))
        return matches

    def _apply_gravity(self):
        for c in range(self.GRID_DIM):
            col = self.grid[:, c]
            non_zeros = col[col != 0]
            new_col = np.zeros_like(col)
            new_col[self.GRID_DIM - len(non_zeros):] = non_zeros
            self.grid[:, c] = new_col

    def _refill_board(self):
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_COLORS + 1)
                    # Sfx: block_fall_sound()

    def _generate_board(self):
        # 1. Generate a stable board with no initial matches
        while True:
            self.grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_DIM, self.GRID_DIM))
            while True:
                matches = self._find_all_matches(self.grid)
                if not matches:
                    break
                for r, c in matches:
                    self.grid[r, c] = 0
                self._apply_gravity()
                self._refill_board()
            
            if np.any(self.grid != 0): break
        
        # 2. Manually insert two "almost-match" setups to guarantee solvability
        for _ in range(2):
            is_horizontal = self.np_random.choice([True, False])
            if is_horizontal:
                r = self.np_random.integers(self.GRID_DIM)
                c = self.np_random.integers(self.GRID_DIM - 3)
                color = self.np_random.integers(1, self.NUM_COLORS + 1)
                self.grid[r, c] = color
                self.grid[r, c + 2] = color
                self.grid[r, c + 3] = color
                while self.grid[r, c+1] == color:
                    self.grid[r, c+1] = self.np_random.integers(1, self.NUM_COLORS + 1)
            else: # Vertical
                r = self.np_random.integers(self.GRID_DIM - 3)
                c = self.np_random.integers(self.GRID_DIM)
                color = self.np_random.integers(1, self.NUM_COLORS + 1)
                self.grid[r, c] = color
                self.grid[r + 2, c] = color
                self.grid[r + 3, c] = color
                while self.grid[r + 1, c] == color:
                    self.grid[r + 1, c] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _check_and_apply_clear_bonus(self, grid_state):
        bonus = 0
        for r in range(self.GRID_DIM):
            if np.all(grid_state[r, :] == 0):
                bonus += self.REWARD_ROW_COL_CLEAR
        for c in range(self.GRID_DIM):
            if np.all(grid_state[:, c] == 0):
                bonus += self.REWARD_ROW_COL_CLEAR
        if bonus > 0: # Sfx: line_clear_sound()
            pass
        return bonus
    
    def _render_game(self):
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                rect = pygame.Rect(
                    self.grid_start_x + c * self.block_size,
                    self.grid_start_y + r * self.block_size,
                    self.block_size, self.block_size,
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

                block_type = self.grid[r, c]
                if block_type != 0:
                    rect_inner = rect.inflate(-4, -4)
                    pygame.draw.rect(self.screen, self.BLOCK_SHADOW_COLORS[block_type], rect_inner, border_radius=6)
                    rect_bevel = rect_inner.inflate(-6, -6)
                    pygame.draw.rect(self.screen, self.BLOCK_COLORS[block_type], rect_bevel, border_radius=4)
        
        if self.blocks_to_flash:
            for r, c in self.blocks_to_flash:
                flash_surface = pygame.Surface((self.block_size, self.block_size), pygame.SRCALPHA)
                flash_surface.fill((*self.COLOR_FLASH, 150))
                self.screen.blit(flash_surface, (self.grid_start_x + c * self.block_size, self.grid_start_y + r * self.block_size))

    def _render_ui(self):
        moves_text = self.font_large.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))

        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_condition = np.all(self.grid == 0)
            msg = "BOARD CLEARED!" if win_condition else "OUT OF MOVES"
            
            end_text = self.font_large.render(msg, True, self.COLOR_FLASH)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            self.screen.blit(end_text, end_rect)
            
            final_score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            self.screen.blit(final_score_text, final_score_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    pygame.display.set_caption("Blockslide Puzzle")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while running:
        action = [0, 0, 0] # Default action: no-op, no buttons
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_r: # Reset game
                    obs, info = env.reset()
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if action[0] != 0: # Only step if a move was made
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_remaining']}, Terminated: {terminated}")

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30)

    env.close()