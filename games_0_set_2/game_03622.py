
# Generated: 2025-08-27T23:54:55.462552
# Source Brief: brief_03622.md
# Brief Index: 3622

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down, turn-based puzzle game where the player changes the color of grid squares
    to create matches of 3 or more. The goal is to clear the entire board within a limited
    number of moves.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to cycle the color of the selected square. "
        "Match 3 or more adjacent squares of the same color to clear them."
    )

    game_description = (
        "A strategic color-matching puzzle. Click adjacent squares to change their colors and match 3 or more "
        "to clear them from the board. Plan your moves to create large combos and clear the board before you run out of moves."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 5
        self.CELL_SIZE = 60
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X_OFFSET = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y_OFFSET = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.MAX_MOVES = 10
        self.NUM_COLORS = 5
        
        # --- Colors ---
        self.COLOR_BG = (25, 28, 36)
        self.COLOR_GRID = (60, 65, 80)
        self.COLOR_EMPTY = (35, 38, 48)
        self.COLORS = [
            (229, 90, 83),   # Red
            (102, 194, 115), # Green
            (83, 142, 247),  # Blue
            (247, 195, 83),  # Yellow
            (178, 102, 247)  # Purple
        ]
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_CURSOR = (255, 255, 255)
        self.COLOR_FLASH = (255, 255, 255)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_large = pygame.font.SysFont('Consolas', 32, bold=True)
            self.font_small = pygame.font.SysFont('Consolas', 24)
        except pygame.error:
            self.font_large = pygame.font.Font(None, 40)
            self.font_small = pygame.font.Font(None, 30)
        
        # --- Game State ---
        self.board = None
        self.cursor_pos = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.cleared_this_step = []
        
        self.reset()
        
        # --- Validation ---
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.board = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.cleared_this_step = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_pressed = action[1] == 1
        
        self.cleared_this_step = []
        reward = 0

        # 1. Handle cursor movement (free action)
        self._move_cursor(movement)

        # 2. Handle color cycle (costs a move)
        if space_pressed:
            self.moves_left -= 1
            # Sfx: a click or 'pop' sound
            
            # Cycle color
            r, c = self.cursor_pos
            current_color = self.board[r, c]
            if current_color > 0: # Cannot change color of empty cell
                self.board[r, c] = (current_color % self.NUM_COLORS) + 1
            
            # Find and clear matches
            matches = self._find_matches((r, c))
            
            if len(matches) >= 3:
                # Sfx: a satisfying 'clear' or 'chime' sound, scaling with match size
                num_cleared = len(matches)
                reward += num_cleared  # +1 for each square cleared
                self.score += num_cleared * 10 # 10 points per square
                
                if num_cleared >= 4:
                    reward += 5  # Bonus for clearing 4+
                    self.score += (num_cleared - 3) * 20 # Combo bonus points
                
                self.cleared_this_step = list(matches)
                for r_m, c_m in matches:
                    self.board[r_m, c_m] = 0 # 0 represents an empty cell

        # 3. Check for termination conditions
        terminated = False
        board_is_clear = np.all(self.board == 0)
        
        if board_is_clear:
            terminated = True
            if not self.game_over:
                reward += 100 # Big win bonus
                self.score += 1000
                # Sfx: a victory fanfare
        elif self.moves_left <= 0:
            terminated = True
            # Sfx: a sad 'fail' sound
        
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _move_cursor(self, movement):
        """Updates cursor position with wraparound."""
        r, c = self.cursor_pos
        if movement == 1: # Up
            r = (r - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2: # Down
            r = (r + 1) % self.GRID_SIZE
        elif movement == 3: # Left
            c = (c - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4: # Right
            c = (c + 1) % self.GRID_SIZE
        
        if (r, c) != self.cursor_pos:
            self.cursor_pos = (r, c)
            # Sfx: a soft 'tick' sound for cursor movement

    def _find_matches(self, start_pos):
        """Finds all orthogonally connected squares of the same color."""
        r_start, c_start = start_pos
        target_color = self.board[r_start, c_start]
        if target_color == 0:
            return set()

        q = deque([start_pos])
        matches = {start_pos}
        
        while q:
            r, c = q.popleft()
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE:
                    if (nr, nc) not in matches and self.board[nr, nc] == target_color:
                        matches.add((nr, nc))
                        q.append((nr, nc))
        
        return matches

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        """Renders the grid, squares, cursor, and effects."""
        # Draw grid squares
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                color_idx = self.board[r, c]
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + c * self.CELL_SIZE,
                    self.GRID_Y_OFFSET + r * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                
                cell_color = self.COLOR_EMPTY if color_idx == 0 else self.COLORS[color_idx - 1]
                pygame.draw.rect(self.screen, cell_color, rect)

        # Draw flash effect for cleared squares
        if self.cleared_this_step:
            flash_surface = self.screen.copy()
            flash_surface.set_colorkey((0,0,0))
            for r, c in self.cleared_this_step:
                rect = pygame.Rect(
                    self.GRID_X_OFFSET + c * self.CELL_SIZE,
                    self.GRID_Y_OFFSET + r * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                pygame.draw.rect(flash_surface, self.COLOR_FLASH, rect)
            flash_surface.set_alpha(150)
            self.screen.blit(flash_surface, (0,0))

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_v = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_v = (self.GRID_X_OFFSET + i * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_v, end_v, 2)
            # Horizontal
            start_h = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            end_h = (self.GRID_X_OFFSET + self.GRID_WIDTH, self.GRID_Y_OFFSET + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_h, end_h, 2)

        # Draw cursor
        r_cursor, c_cursor = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + c_cursor * self.CELL_SIZE,
            self.GRID_Y_OFFSET + r_cursor * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4)

    def _render_ui(self):
        """Renders the score and remaining moves."""
        # Moves Left
        moves_text = f"MOVES: {self.moves_left}"
        moves_surf = self.font_large.render(moves_text, True, self.COLOR_TEXT)
        self.screen.blit(moves_surf, (20, 10))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_large.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(score_surf, score_rect)
        
        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            win = np.all(self.board == 0)
            msg = "BOARD CLEARED!" if win else "OUT OF MOVES"
            msg_surf = self.font_large.render(msg, True, self.COLOR_FLASH)
            msg_rect = msg_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            
            overlay.blit(msg_surf, msg_rect)
            self.screen.blit(overlay, (0,0))

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "steps": 0, # Steps are not tracked as it's turn-based
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Use a separate display for human play
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Color Match Puzzle")
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("\n" + "="*30)
    print("      Color Match Puzzle")
    print("="*30 + "\n")
    print(env.user_guide)
    
    running = True
    while running:
        # Reset action for this frame
        action.fill(0)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if terminated:
                    continue
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                
                # Only step if a key was pressed, since auto_advance is False
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")

        # Update display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()