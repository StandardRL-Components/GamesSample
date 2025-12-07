
# Generated: 2025-08-27T14:12:59.962367
# Source Brief: brief_00612.md
# Brief Index: 612

        
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
        "Controls: Arrow keys to move cursor. Space to reveal a square. Shift to place/remove a flag."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist Minesweeper clone. Reveal all safe squares to win, but avoid the mines."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array", grid_size=(10, 10), num_mines=15):
        super().__init__()

        # Game parameters
        self.grid_w, self.grid_h = grid_size
        self.num_mines = num_mines
        self.max_steps = self.grid_w * self.grid_h * 2 # Generous step limit

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        try:
            self.font_main = pygame.font.Font(None, 36)
            self.font_grid = pygame.font.Font(None, 28)
        except IOError: # Fallback if default font not found
            self.font_main = pygame.font.SysFont("sans", 36)
            self.font_grid = pygame.font.SysFont("sans", 28)

        # Colors
        self.COLOR_BG = (25, 28, 32)
        self.COLOR_GRID = (60, 65, 70)
        self.COLOR_UNREVEALED = (45, 50, 55)
        self.COLOR_REVEALED = (80, 88, 96)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_FLAG = (240, 180, 0)
        self.COLOR_MINE = (220, 50, 50)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLOR_NUMBERS = [
            self.COLOR_REVEALED,
            (50, 150, 255),  # 1
            (50, 200, 100),  # 2
            (255, 100, 100),  # 3
            (150, 50, 255),   # 4
            (255, 150, 50),   # 5
            (50, 200, 200),   # 6
            (200, 50, 200),   # 7
            (100, 100, 100)   # 8
        ]

        # Game state variables (initialized in reset)
        self.mine_grid = None
        self.number_grid = None
        self.revealed_grid = None
        self.flagged_grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.num_safe_squares_revealed = 0
        self.num_correct_flags = 0

        self.reset()
        
        # Self-validation check
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize grids
        self.mine_grid = np.zeros((self.grid_w, self.grid_h), dtype=bool)
        self.number_grid = np.zeros((self.grid_w, self.grid_h), dtype=int)
        self.revealed_grid = np.zeros((self.grid_w, self.grid_h), dtype=bool)
        self.flagged_grid = np.zeros((self.grid_w, self.grid_h), dtype=bool)
        
        # Place mines and calculate numbers
        self._place_mines()
        self._calculate_numbers()

        self.total_safe_squares = self.grid_w * self.grid_h - self.num_mines
        self.num_safe_squares_revealed = 0
        self.num_correct_flags = 0

        # Reset game state
        self.cursor_pos = [self.grid_w // 2, self.grid_h // 2]
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        return self._get_observation(), self._get_info()

    def _place_mines(self):
        # Flatten the grid indices to randomly choose mine locations
        flat_indices = list(range(self.grid_w * self.grid_h))
        mine_indices = self.np_random.choice(flat_indices, self.num_mines, replace=False)
        for idx in mine_indices:
            x = idx % self.grid_w
            y = idx // self.grid_w
            self.mine_grid[x, y] = True

    def _calculate_numbers(self):
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                if not self.mine_grid[x, y]:
                    count = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and self.mine_grid[nx, ny]:
                                count += 1
                    self.number_grid[x, y] = count

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        reward = 0.0

        # 1. Process movement
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.grid_h) % self.grid_h
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.grid_h
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.grid_w) % self.grid_w
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.grid_w
        
        cx, cy = self.cursor_pos
        
        # 2. Process actions (Reveal has priority over Flag)
        if space_held: # Reveal
            if not self.revealed_grid[cx, cy] and not self.flagged_grid[cx, cy]:
                if self.mine_grid[cx, cy]:
                    # Game Over: Hit a mine
                    self.game_over = True
                    self.win = False
                    reward = -100.0
                    # SFX: Explosion
                else:
                    # Revealed a safe square
                    revealed_count = self._reveal_square(cx, cy)
                    reward += revealed_count # +1 for each newly revealed safe square
            else:
                # Wasted action
                reward -= 0.5

        elif shift_held: # Flag/Unflag
            if not self.revealed_grid[cx, cy]:
                was_flagged = self.flagged_grid[cx, cy]
                self.flagged_grid[cx, cy] = not was_flagged
                
                # SFX: Flag place/remove
                
                is_mine = self.mine_grid[cx, cy]
                
                if not was_flagged: # Placing a flag
                    reward -= 0.1 # Small penalty for using a flag
                    if is_mine:
                        reward += 10.0 # Big bonus for correct flag
                        self.num_correct_flags += 1
                else: # Removing a flag
                    if is_mine:
                        reward -= 10.0 # Penalty for removing correct flag
                        self.num_correct_flags -= 1
            else:
                # Wasted action
                reward -= 0.5

        # 3. Check for win condition
        if not self.game_over and self.num_safe_squares_revealed == self.total_safe_squares:
            self.game_over = True
            self.win = True
            reward += 100.0 # Win bonus
            # SFX: Victory fanfare

        # 4. Check for termination by steps
        terminated = self.game_over or self.steps >= self.max_steps
        if self.steps >= self.max_steps and not self.game_over:
            reward -= 50 # Penalty for running out of time

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _reveal_square(self, x, y):
        # Use a stack for an iterative flood fill
        stack = [(x, y)]
        visited = set()
        revealed_count = 0

        while stack:
            cx, cy = stack.pop()
            
            if (cx, cy) in visited:
                continue
            visited.add((cx,cy))

            if not (0 <= cx < self.grid_w and 0 <= cy < self.grid_h):
                continue
            if self.revealed_grid[cx, cy] or self.flagged_grid[cx, cy]:
                continue
            
            self.revealed_grid[cx, cy] = True
            revealed_count += 1
            self.num_safe_squares_revealed += 1
            # SFX: Tile reveal
            
            # If the square is a 0, add its neighbors to the stack to be revealed
            if self.number_grid[cx, cy] == 0:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        stack.append((cx + dx, cy + dy))
        
        return revealed_count

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Calculate grid rendering properties
        grid_pixel_h = 360
        cell_size = grid_pixel_h // self.grid_h
        grid_pixel_w = cell_size * self.grid_w
        
        start_x = (640 - grid_pixel_w) // 2
        start_y = (400 - grid_pixel_h) // 2

        for x in range(self.grid_w):
            for y in range(self.grid_h):
                rect = pygame.Rect(start_x + x * cell_size, start_y + y * cell_size, cell_size, cell_size)
                
                # Determine cell content and color
                is_revealed = self.revealed_grid[x, y]
                is_flagged = self.flagged_grid[x, y]
                is_mine = self.mine_grid[x, y]
                
                # Draw cell background
                bg_color = self.COLOR_UNREVEALED
                if is_revealed or (self.game_over and is_mine):
                    bg_color = self.COLOR_REVEALED
                
                pygame.draw.rect(self.screen, bg_color, rect)
                
                # Draw content
                if self.game_over and is_mine and not is_flagged:
                    pygame.gfxdraw.filled_circle(self.screen, rect.centerx, rect.centery, int(cell_size * 0.3), self.COLOR_MINE)
                    pygame.gfxdraw.aacircle(self.screen, rect.centerx, rect.centery, int(cell_size * 0.3), self.COLOR_MINE)
                elif is_flagged:
                    # Draw a flag
                    flag_poly = [
                        (rect.centerx - int(cell_size*0.1), rect.centery + int(cell_size*0.3)),
                        (rect.centerx - int(cell_size*0.1), rect.centery - int(cell_size*0.3)),
                        (rect.centerx + int(cell_size*0.3), rect.centery - int(cell_size*0.1)),
                        (rect.centerx - int(cell_size*0.1), rect.centery + int(cell_size*0.1)),
                    ]
                    pygame.draw.polygon(self.screen, self.COLOR_FLAG, flag_poly)
                    if self.game_over and not is_mine: # Incorrect flag
                        pygame.draw.line(self.screen, self.COLOR_MINE, rect.topleft, rect.bottomright, 2)
                        pygame.draw.line(self.screen, self.COLOR_MINE, rect.topright, rect.bottomleft, 2)

                elif is_revealed:
                    num = self.number_grid[x, y]
                    if num > 0:
                        num_surf = self.font_grid.render(str(num), True, self.COLOR_NUMBERS[num])
                        num_rect = num_surf.get_rect(center=rect.center)
                        self.screen.blit(num_surf, num_rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(start_x + cx * cell_size, start_y + cy * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # Flags remaining
        flags_placed = np.sum(self.flagged_grid)
        flags_text = f"MINES: {self.num_mines - flags_placed}"
        flags_surf = self.font_main.render(flags_text, True, self.COLOR_TEXT)
        self.screen.blit(flags_surf, (20, 5))

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_main.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(centerx=640 // 2)
        score_rect.top = 5
        self.screen.blit(score_surf, score_rect)

        # Steps
        steps_text = f"STEPS: {self.steps}"
        steps_surf = self.font_main.render(steps_text, True, self.COLOR_TEXT)
        steps_rect = steps_surf.get_rect(right=620)
        steps_rect.top = 5
        self.screen.blit(steps_surf, steps_rect)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            end_text = "YOU WIN!" if self.win else "GAME OVER"
            end_color = (100, 255, 100) if self.win else self.COLOR_MINE
            
            end_surf = self.font_main.render(end_text, True, end_color)
            end_rect = end_surf.get_rect(center=(640 // 2, 400 // 2))
            self.screen.blit(end_surf, end_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": self.cursor_pos,
            "flags_placed": np.sum(self.flagged_grid),
            "correct_flags": self.num_correct_flags,
            "safe_squares_revealed": self.num_safe_squares_revealed,
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
        assert test_obs.shape == (400, 640, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        assert 'score' in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert 'score' in info

        # Test specific game mechanics
        # Assert number of mines
        self.reset()
        assert np.sum(self.mine_grid) == self.num_mines
        # Assert number calculation
        for x in range(self.grid_w):
            for y in range(self.grid_h):
                if not self.mine_grid[x, y]:
                    count = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0: continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h and self.mine_grid[nx, ny]:
                                count += 1
                    assert self.number_grid[x, y] == count, f"Number mismatch at ({x},{y})"

        print("✓ Implementation validated successfully")

# Example of how to run the environment for interactive play
if __name__ == '__main__':
    import sys

    # Map keys to actions for human play
    key_to_action = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
        pygame.K_SPACE: [0, 1, 0],
        pygame.K_LSHIFT: [0, 0, 1],
        pygame.K_RSHIFT: [0, 0, 1],
    }

    env = GameEnv(grid_size=(10, 10), num_mines=15)
    
    # Set up a visible display
    pygame.display.set_caption("Minesweeper Gym Environment")
    visible_screen = pygame.display.set_mode((640, 400))

    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    while True:
        action = [0, 0, 0] # Default no-op action

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    done = False
                    print("\n--- Game Reset ---")
                    continue
                if event.key in key_to_action:
                    action = key_to_action[event.key]
        
        # Step the environment only if an action is taken and the game is not over
        if any(action) and not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")
            if done:
                print("--- GAME OVER ---")
                print(f"Final Score: {info['score']:.2f}, Win: {info['win']}")

        # Render the current observation to the visible screen
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        visible_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS