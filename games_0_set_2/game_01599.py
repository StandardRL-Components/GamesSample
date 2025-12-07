
# Generated: 2025-08-28T02:06:45.028381
# Source Brief: brief_01599.md
# Brief Index: 1599

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to shift the selected color block. "
        "Space/Shift to cycle through colors."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A pixel-art puzzle. Shift blocks of color to match the target image "
        "before you run out of moves. All blocks of a selected color move together."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 12
        self.BLOCK_SIZE = 24
        self.GRID_LINE_WIDTH = 2
        self.MAX_MOVES = 20
        self.SCRAMBLE_STEPS = 8

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_BG = (30, 35, 55)
        self.COLOR_UI_BG = (40, 45, 65)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_VALUE = (255, 255, 100)
        self.COLOR_HIGHLIGHT = (255, 255, 255, 100) # RGBA

        # Grid cell values and their corresponding colors
        self.EMPTY = 0
        self.OBSTACLE = 1
        self.RED = 2
        self.GREEN = 3
        self.BLUE = 4
        self.YELLOW = 5
        self.COLORS = {
            self.EMPTY: self.COLOR_GRID_BG,
            self.OBSTACLE: (80, 80, 90),
            self.RED: (220, 50, 50),
            self.GREEN: (50, 220, 50),
            self.BLUE: (50, 100, 220),
            self.YELLOW: (220, 220, 50),
        }
        self.MOVABLE_COLORS = [self.RED, self.GREEN, self.BLUE, self.YELLOW]

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_s = pygame.font.SysFont("monospace", 16)
        self.font_m = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_l = pygame.font.SysFont("monospace", 32, bold=True)
        
        # --- EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Grid Layout ---
        self.grid_pixel_width = self.GRID_WIDTH * self.BLOCK_SIZE
        self.grid_pixel_height = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.grid_x = 50
        self.grid_y = (self.SCREEN_HEIGHT - self.grid_pixel_height) // 2

        self.ui_x = self.grid_x + self.grid_pixel_width + 50
        self.ui_width = self.SCREEN_WIDTH - self.ui_x - 30

        # --- State Variables ---
        self.target_grid = self._create_target_grid()
        self.current_grid = np.zeros_like(self.target_grid)
        self.moves_left = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.selected_color_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_reward_text = ""

        # Initialize state by calling reset
        self.reset()
        
        # Self-check
        self.validate_implementation()

    def _create_target_grid(self):
        grid = np.full((self.GRID_HEIGHT, self.GRID_WIDTH), self.EMPTY, dtype=np.int32)
        # Smiley face pattern
        grid[2:4, 4:8] = self.YELLOW # Top hair
        grid[4:9, 3:9] = self.YELLOW # Face
        grid[5, 4] = self.BLUE      # Left eye
        grid[5, 7] = self.BLUE      # Right eye
        grid[7, 4:8] = self.RED       # Mouth
        # Obstacles
        grid[0, :] = self.OBSTACLE
        grid[:, 0] = self.OBSTACLE
        grid[self.GRID_HEIGHT-1, :] = self.OBSTACLE
        grid[:, self.GRID_WIDTH-1] = self.OBSTACLE
        grid[10, 4:8] = self.OBSTACLE
        return grid

    def _scramble_grid(self, grid):
        scrambled = np.copy(grid)
        original_grid_state = np.copy(self.current_grid)
        
        for _ in range(self.SCRAMBLE_STEPS):
            temp_grid = np.copy(scrambled)
            
            # Pick a random movable color and a random direction
            color_val = self.np_random.choice(self.MOVABLE_COLORS)
            direction_idx = self.np_random.integers(1, 5) # 1-4 for directions
            
            # Temporarily set `self.current_grid` to apply the shift
            self.current_grid = temp_grid
            if self._shift_blocks(direction_idx, color_val):
                scrambled = self.current_grid
        
        # Restore original grid state for the env
        self.current_grid = original_grid_state
        return scrambled

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_grid = self._scramble_grid(self.target_grid)
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.selected_color_index = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_reward_text = ""
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0

        # 1. Handle color selection (does not consume a move)
        if space_held and not self.prev_space_held:
            self.selected_color_index = (self.selected_color_index + 1) % len(self.MOVABLE_COLORS)
            # sfx: UI_CYCLE_SOUND
        if shift_held and not self.prev_shift_held:
            self.selected_color_index = (self.selected_color_index - 1 + len(self.MOVABLE_COLORS)) % len(self.MOVABLE_COLORS)
            # sfx: UI_CYCLE_SOUND

        # 2. Handle block shifting (consumes a move)
        if movement != 0:
            old_match_score = np.sum(self.current_grid == self.target_grid)
            
            selected_color_val = self.MOVABLE_COLORS[self.selected_color_index]
            if self._shift_blocks(movement, selected_color_val):
                self.moves_left -= 1
                # sfx: BLOCK_SLIDE_SOUND
                
                new_match_score = np.sum(self.current_grid == self.target_grid)
                pixel_change_reward = new_match_score - old_match_score
                reward += pixel_change_reward
                self.last_reward_text = f"{pixel_change_reward:+}"
            else:
                # sfx: BUMP_SOUND
                reward -= 0.1 # Small penalty for invalid move attempt
                self.last_reward_text = "INVALID"

        self.steps += 1
        self.score += reward

        # 3. Check for termination
        is_solved = np.array_equal(self.current_grid, self.target_grid)
        is_out_of_moves = self.moves_left <= 0
        terminated = is_solved or is_out_of_moves
        
        if terminated:
            self.game_over = True
            if is_solved:
                reward += 100
                self.score += 100
                self.last_reward_text = "SOLVED! +100"
                # sfx: PUZZLE_SOLVED_JINGLE
            else: # Out of moves
                reward -= 50
                self.score -= 50
                self.last_reward_text = "FAILED! -50"
                # sfx: GAME_OVER_SOUND

        # Update button states for next step
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _shift_blocks(self, direction_idx, color_val):
        # direction_idx: 1=up, 2=down, 3=left, 4=right
        if direction_idx == 1: dy, dx = -1, 0
        elif direction_idx == 2: dy, dx = 1, 0
        elif direction_idx == 3: dy, dx = 0, -1
        elif direction_idx == 4: dy, dx = 0, 1
        else: return False

        moving_cells = np.argwhere(self.current_grid == color_val)
        if len(moving_cells) == 0:
            return False

        # Check if the move is valid for all blocks of the selected color
        for r, c in moving_cells:
            nr, nc = r + dy, c + dx
            # Check bounds
            if not (0 <= nr < self.GRID_HEIGHT and 0 <= nc < self.GRID_WIDTH):
                return False # Hits edge
            # Check destination cell
            dest_val = self.current_grid[nr, nc]
            if dest_val != self.EMPTY and dest_val != color_val:
                return False # Hits another color block or obstacle

        # If valid, perform the move, sorted to prevent self-overwriting
        if dy == 1: moving_cells = sorted(moving_cells, key=lambda p: p[0], reverse=True) # Down
        elif dy == -1: moving_cells = sorted(moving_cells, key=lambda p: p[0]) # Up
        elif dx == 1: moving_cells = sorted(moving_cells, key=lambda p: p[1], reverse=True) # Right
        elif dx == -1: moving_cells = sorted(moving_cells, key=lambda p: p[1]) # Left

        # Create a temporary copy to modify
        new_grid = np.copy(self.current_grid)
        
        # First, clear old positions
        for r, c in moving_cells:
            new_grid[r, c] = self.EMPTY
        
        # Then, fill new positions
        for r, c in moving_cells:
            nr, nc = r + dy, c + dx
            new_grid[nr, nc] = color_val

        self.current_grid = new_grid
        return True

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
            "moves_left": self.moves_left,
            "is_solved": np.array_equal(self.current_grid, self.target_grid),
        }
        
    def _render_grid(self, grid, x_offset, y_offset, block_size, title=""):
        grid_h, grid_w = grid.shape
        pixel_w, pixel_h = grid_w * block_size, grid_h * block_size
        
        # Grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (x_offset, y_offset, pixel_w, pixel_h))

        # Render blocks
        for r in range(grid_h):
            for c in range(grid_w):
                color_val = grid[r, c]
                if color_val != self.EMPTY:
                    color = self.COLORS.get(color_val, (255, 0, 255))
                    rect = (
                        x_offset + c * block_size,
                        y_offset + r * block_size,
                        block_size,
                        block_size,
                    )
                    pygame.draw.rect(self.screen, color, rect)

        # Grid lines
        for i in range(grid_w + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (x_offset + i * block_size, y_offset), (x_offset + i * block_size, y_offset + pixel_h), self.GRID_LINE_WIDTH)
        for i in range(grid_h + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (x_offset, y_offset + i * block_size), (x_offset + pixel_w, y_offset + i * block_size), self.GRID_LINE_WIDTH)

        if title:
            title_surf = self.font_m.render(title, True, self.COLOR_TEXT)
            self.screen.blit(title_surf, (x_offset, y_offset - 25))

    def _render_game(self):
        # Render main grid
        self._render_grid(self.current_grid, self.grid_x, self.grid_y, self.BLOCK_SIZE, "CURRENT STATE")
        
        # Highlight selected color blocks
        selected_color_val = self.MOVABLE_COLORS[self.selected_color_index]
        highlight_surface = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
        highlight_surface.fill(self.COLOR_HIGHLIGHT)
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.current_grid[r, c] == selected_color_val:
                    self.screen.blit(highlight_surface, (self.grid_x + c * self.BLOCK_SIZE, self.grid_y + r * self.BLOCK_SIZE))

    def _render_ui(self):
        # UI Panel background
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (self.ui_x, 0, self.ui_width, self.SCREEN_HEIGHT))

        y_pos = 30

        # Target Image
        self._render_grid(self.target_grid, self.ui_x + 20, y_pos, block_size=10, title="TARGET")
        y_pos += self.GRID_HEIGHT * 10 + 50

        # Moves Left
        moves_text = self.font_m.render("MOVES LEFT", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.ui_x + 20, y_pos))
        moves_val_text = self.font_l.render(str(self.moves_left), True, self.COLOR_TEXT_VALUE)
        self.screen.blit(moves_val_text, (self.ui_x + 20, y_pos + 25))
        y_pos += 70

        # Score
        score_text = self.font_m.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.ui_x + 20, y_pos))
        score_val_text = self.font_l.render(f"{self.score:.0f}", True, self.COLOR_TEXT_VALUE)
        self.screen.blit(score_val_text, (self.ui_x + 20, y_pos + 25))
        
        if self.last_reward_text:
            reward_surf = self.font_s.render(self.last_reward_text, True, self.COLOR_TEXT_VALUE)
            self.screen.blit(reward_surf, (self.ui_x + 20 + score_val_text.get_width() + 10, y_pos + 40))
        y_pos += 70

        # Selected Color
        sel_text = self.font_m.render("SELECTED", True, self.COLOR_TEXT)
        self.screen.blit(sel_text, (self.ui_x + 20, y_pos))
        
        selected_color_val = self.MOVABLE_COLORS[self.selected_color_index]
        color_swatch = self.COLORS.get(selected_color_val)
        pygame.draw.rect(self.screen, color_swatch, (self.ui_x + 20, y_pos + 30, 50, 50))
        pygame.draw.rect(self.screen, self.COLOR_TEXT, (self.ui_x + 20, y_pos + 30, 50, 50), 2)
        
        # Game Over Message
        if self.game_over:
            is_solved = np.array_equal(self.current_grid, self.target_grid)
            msg = "PUZZLE SOLVED!" if is_solved else "OUT OF MOVES"
            color = (100, 255, 100) if is_solved else (255, 100, 100)
            
            overlay = pygame.Surface((self.grid_pixel_width, 100), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            msg_surf = self.font_l.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.grid_pixel_width // 2, 50))
            overlay.blit(msg_surf, msg_rect)
            
            self.screen.blit(overlay, (self.grid_x, self.grid_y + self.grid_pixel_height // 2 - 50))


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

# Example of how to run the environment
if __name__ == '__main__':
    import os
    # This check needs to run headlessly.
    # To play manually, comment this line out and install pygame.
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    env = GameEnv()
    
    # --- Manual Play Example (requires a display) ---
    # To run this part, comment out the os.environ line above
    # and make sure you have a display environment.
    #
    # os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "quartz"
    # pygame.display.init()
    # pygame.font.init()
    # screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    # obs, info = env.reset()
    # done = False
    # clock = pygame.time.Clock()
    
    # print(GameEnv.user_guide)
    # print(GameEnv.game_description)

    # while not done:
    #     action = [0, 0, 0] # no-op, released, released
        
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             done = True

    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_UP]: action[0] = 1
    #     elif keys[pygame.K_DOWN]: action[0] = 2
    #     elif keys[pygame.K_LEFT]: action[0] = 3
    #     elif keys[pygame.K_RIGHT]: action[0] = 4
        
    #     if keys[pygame.K_SPACE]: action[1] = 1
    #     if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1
        
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     done = terminated or truncated

    #     # Draw the observation from the environment to the screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
        
    #     clock.tick(10) # Limit FPS for human playability
        
    #     if reward != 0:
    #         print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves Left: {info['moves_left']}")
    #     if done:
    #         print("Game Over!")
    #         pygame.time.wait(2000)

    # env.close()