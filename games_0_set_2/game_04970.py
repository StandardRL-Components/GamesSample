
# Generated: 2025-08-28T03:34:30.599656
# Source Brief: brief_04970.md
# Brief Index: 4970

        
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
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to toggle a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a procedurally generated minefield, carefully revealing safe squares while avoiding hidden mines to clear the board."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    GRID_SIZE = 5
    NUM_MINES = 10
    MAX_STEPS = 1000
    
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (40, 42, 54)
    COLOR_GRID_LINE = (30, 32, 42)
    COLOR_UNREVEALED = (98, 114, 164)
    COLOR_REVEALED = (220, 220, 220)
    COLOR_CURSOR = (241, 250, 140)
    COLOR_FLAG = (80, 250, 123)
    COLOR_MINE_BG = (255, 85, 85)
    COLOR_MINE_FG = (20, 20, 20)
    COLOR_TEXT = (248, 248, 242)
    
    NUMBER_COLORS = {
        1: (98, 160, 234),   # Blue
        2: (80, 250, 123),   # Green
        3: (255, 85, 85),    # Red
        4: (68, 71, 90),     # Dark Blue
        5: (189, 147, 249),  # Purple
        6: (139, 233, 253),  # Cyan
        7: (20, 20, 20),     # Black
        8: (150, 150, 150),  # Grey
    }

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
        self.font_small = pygame.font.Font(None, 24)
        
        # Grid layout
        self.cell_size = 60
        self.grid_width = self.GRID_SIZE * self.cell_size
        self.grid_height = self.GRID_SIZE * self.cell_size
        self.grid_offset_x = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_offset_y = (self.SCREEN_HEIGHT - self.grid_height) // 2 + 20

        # Initialize state variables
        self.grid = None
        self.revealed_mask = None
        self.flagged_mask = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.explosion_pos = None
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.explosion_pos = None
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self._generate_grid()
        self.revealed_mask = np.full((self.GRID_SIZE, self.GRID_SIZE), False, dtype=bool)
        self.flagged_mask = np.full((self.GRID_SIZE, self.GRID_SIZE), False, dtype=bool)
        
        self.safe_squares_total = self.GRID_SIZE * self.GRID_SIZE - self.NUM_MINES

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        
        # Place mines (-1)
        mine_indices = self.np_random.choice(self.GRID_SIZE * self.GRID_SIZE, self.NUM_MINES, replace=False)
        for idx in mine_indices:
            row, col = np.unravel_index(idx, (self.GRID_SIZE, self.GRID_SIZE))
            self.grid[row, col] = -1
            
        # Calculate numbers
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == -1:
                    continue
                mine_count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and self.grid[nr, nc] == -1:
                            mine_count += 1
                self.grid[r, c] = mine_count

    def step(self, action):
        reward = 0
        
        if self.game_over or self.win:
            return self._get_observation(), 0, True, False, self._get_info()
            
        movement, space_press, shift_press = action[0], action[1] == 1, action[2] == 1

        # 1. Handle cursor movement
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_SIZE
        
        # 2. Handle actions (mutually exclusive for clarity)
        cy, cx = self.cursor_pos[1], self.cursor_pos[0]
        
        if shift_press:
            # Toggle flag
            if not self.revealed_mask[cy, cx]:
                self.flagged_mask[cy, cx] = not self.flagged_mask[cy, cx]
                reward = -0.2 # Small penalty for using a flag
                # sound: flag_place.wav or flag_remove.wav
        
        elif space_press:
            # Reveal square
            if not self.revealed_mask[cy, cx] and not self.flagged_mask[cy, cx]:
                revealed_count = self._reveal_square(cy, cx)
                if revealed_count > 0:
                    reward = revealed_count # +1 for each new safe square
                    # sound: reveal.wav
                
                if self.grid[cy, cx] == -1:
                    self.game_over = True
                    self.explosion_pos = (cx, cy)
                    # sound: explosion.wav
        
        self.steps += 1
        self.score += reward
        
        terminated = self._check_termination()
        if terminated and not self.game_over: # Win condition
            self.win = True
            reward = 100
            self.score += reward
            # sound: win.wav
        elif terminated and self.game_over: # Loss condition
            reward = -100
            self.score += reward
            
        if self.steps >= self.MAX_STEPS:
            terminated = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _reveal_square(self, r, c):
        if self.revealed_mask[r, c]:
            return 0
        
        self.revealed_mask[r, c] = True
        
        if self.grid[r, c] == -1:
            return 0 # Hit a mine
            
        revealed_count = 1
        
        # Flood fill if it's an empty square
        if self.grid[r, c] == 0:
            q = deque([(r, c)])
            while q:
                curr_r, curr_c = q.popleft()
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and \
                           not self.revealed_mask[nr, nc] and not self.flagged_mask[nr, nc]:
                            
                            self.revealed_mask[nr, nc] = True
                            revealed_count += 1
                            if self.grid[nr, nc] == 0:
                                q.append((nr, nc))
        return revealed_count

    def _check_termination(self):
        if self.game_over:
            return True
        safe_squares_revealed = np.sum(self.revealed_mask)
        if safe_squares_revealed == self.safe_squares_total:
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.grid_offset_x + c * self.cell_size,
                    self.grid_offset_y + r * self.cell_size,
                    self.cell_size, self.cell_size
                )

                # Draw cell background
                if self.game_over and self.grid[r, c] == -1:
                    is_exploded_mine = self.explosion_pos and self.explosion_pos == (c, r)
                    pygame.draw.rect(self.screen, self.COLOR_MINE_BG if is_exploded_mine else self.COLOR_UNREVEALED, rect)
                elif self.revealed_mask[r, c]:
                    pygame.draw.rect(self.screen, self.COLOR_REVEALED, rect)
                else:
                    pygame.draw.rect(self.screen, self.COLOR_UNREVEALED, rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)

                # Draw cell content
                if self.game_over and self.grid[r, c] == -1:
                    # Draw mine icon
                    center_x, center_y = rect.center
                    pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.cell_size // 4, self.COLOR_MINE_FG)
                elif self.revealed_mask[r, c] and self.grid[r, c] > 0:
                    # Draw number
                    num_text = str(self.grid[r, c])
                    color = self.NUMBER_COLORS.get(self.grid[r, c], self.COLOR_TEXT)
                    text_surf = self.font_large.render(num_text, True, color)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)
                elif self.flagged_mask[r, c] and not self.revealed_mask[r, c]:
                    # Draw flag
                    center_x, center_y = rect.center
                    flag_points = [
                        (center_x - 10, center_y - 15),
                        (center_x + 10, center_y - 10),
                        (center_x - 10, center_y - 5)
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_points)
                    pygame.draw.line(self.screen, self.COLOR_FLAG, (center_x - 10, center_y - 15), (center_x - 10, center_y + 15), 3)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos[1], self.cursor_pos[0]
        cursor_rect = pygame.Rect(
            self.grid_offset_x + cursor_c * self.cell_size,
            self.grid_offset_y + cursor_r * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4, border_radius=4)
        
    def _render_ui(self):
        # Score display
        score_text = f"Score: {self.score:.1f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 10))
        
        # Steps display
        steps_text = f"Steps: {self.steps}/{self.MAX_STEPS}"
        steps_surf = self.font_small.render(steps_text, True, self.COLOR_TEXT)
        steps_rect = steps_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(steps_surf, steps_rect)
        
        # Game over/win message
        message = ""
        if self.win:
            message = "BOARD CLEARED!"
        elif self.game_over:
            message = "GAME OVER"
            
        if message:
            msg_surf = self.font_large.render(message, True, self.COLOR_CURSOR)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.grid_offset_y - 25))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "win": self.win
        }

    def close(self):
        pygame.quit()

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

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Minesweeper Gym Env")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    print("--- Manual Control ---")
    print(env.user_guide)
    print("Q to quit.")
    
    running = True
    while running:
        # Reset action for this frame
        action.fill(0)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            # Movement
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            else: action[0] = 0

            # Actions
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
            
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(10) # Limit frame rate for human playability

    env.close()