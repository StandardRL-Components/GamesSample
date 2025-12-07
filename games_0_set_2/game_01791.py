
# Generated: 2025-08-28T02:44:06.446544
# Source Brief: brief_01791.md
# Brief Index: 1791

        
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
        "Controls: Arrow keys to move the cursor. Press space to paint the selected square and its neighbors."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A turn-based puzzle game. Paint squares on the grid to match the target patterns before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH - 160) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2

        self.MAX_MOVES = 20
        self.NUM_PATTERNS = 5
        self.PATTERN_COMPLEXITY = 3 # Number of paint actions to generate a pattern

        # --- Colors ---
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID_LINE = (50, 50, 60)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_FLASH = (255, 255, 255)

        # 0=Unpainted, 1-5=Paintable colors
        self.COLORS = [
            (80, 80, 90),       # 0: Gray (Unpainted)
            (220, 50, 50),      # 1: Red
            (50, 220, 50),      # 2: Green
            (50, 100, 220),     # 3: Blue
            (220, 220, 50),     # 4: Yellow
            (150, 50, 220),     # 5: Purple
        ]
        self.NUM_PAINT_COLORS = len(self.COLORS) - 1

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
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        # --- State Variables (initialized in reset) ---
        self.grid = None
        self.target_patterns = None
        self.cursor_pos = None
        self.moves_left = None
        self.score = None
        self.current_level = None
        self.game_over = None
        self.steps = None
        self.flash_effects = None

    def _paint_cell(self, grid, x, y):
        """Applies the color-cycling paint logic to a single cell."""
        if 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE:
            current_color_idx = grid[y, x]
            # Cycle through paintable colors (1 to N), mapping 0 to 1.
            new_color_idx = (current_color_idx % self.NUM_PAINT_COLORS) + 1
            grid[y, x] = new_color_idx

    def _apply_paint_action(self, grid, x, y):
        """Applies the paint logic to the target cell and its neighbors."""
        self._paint_cell(grid, x, y)
        self._paint_cell(grid, x + 1, y)
        self._paint_cell(grid, x - 1, y)
        self._paint_cell(grid, x, y + 1)
        self._paint_cell(grid, x, y - 1)
        
    def _generate_patterns(self):
        """Generates a set of solvable target patterns."""
        self.target_patterns = []
        for _ in range(self.NUM_PATTERNS):
            pattern = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
            # Use self.np_random for reproducibility
            num_actions = self.np_random.integers(2, self.PATTERN_COMPLEXITY + 2)
            for _ in range(num_actions):
                px = self.np_random.integers(0, self.GRID_SIZE)
                py = self.np_random.integers(0, self.GRID_SIZE)
                self._apply_paint_action(pattern, px, py)
            self.target_patterns.append(pattern)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_patterns()
        
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.current_level = 0
        self.game_over = False
        self.steps = 0
        self.flash_effects = [] # List of (pos, timer) for visual effects

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        self.steps += 1
        reward = 0
        
        # --- Handle Input ---
        # Cursor movement does not consume a turn in this turn-based game
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE

        # --- Handle Paint Action (consumes a turn) ---
        if space_held and not self.game_over:
            self.moves_left -= 1
            
            # Apply paint
            paint_x, paint_y = self.cursor_pos
            self._apply_paint_action(self.grid, paint_x, paint_y)
            # Add flash effect for visual feedback
            # sound: "paint_splash.wav"
            self.flash_effects.append(((paint_x, paint_y), 5))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                if 0 <= paint_x + dx < self.GRID_SIZE and 0 <= paint_y + dy < self.GRID_SIZE:
                    self.flash_effects.append(((paint_x + dx, paint_y + dy), 5))
            
            reward = self._calculate_reward()

        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _calculate_reward(self):
        """Calculates reward based on current game state after a paint action."""
        reward = 0
        # Check for Pattern Completion
        current_target = self.target_patterns[self.current_level]
        if np.array_equal(self.grid, current_target):
            # Pattern complete!
            # sound: "level_complete.wav"
            reward += 10.0
            self.score += 10
            self.current_level += 1
            self.grid.fill(0) # Reset grid for next level
            
            if self.current_level >= self.NUM_PATTERNS:
                # All patterns complete - WIN
                # sound: "game_win.wav"
                reward += 50.0
                self.score += 50
        else:
            # Continuous Reward Calculation
            matches = np.sum(self.grid == current_target)
            total_cells = self.GRID_SIZE * self.GRID_SIZE
            mismatches = total_cells - matches
            # Continuous reward for partial progress
            reward += (matches / total_cells) * 1.0
            reward -= (mismatches / total_cells) * 0.1
        return reward
    
    def _check_termination(self):
        """Checks if the episode should terminate."""
        if self.current_level >= self.NUM_PATTERNS:
            self.game_over = True
        elif self.moves_left <= 0:
            # sound: "game_over.wav"
            self.game_over = True
        elif self.steps >= 1000: # Fallback step limit
            self.game_over = True
        
        return self.game_over

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
            "level": self.current_level,
        }

    def _render_game(self):
        # --- Draw Grid ---
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(
                    self.GRID_X + x * self.CELL_SIZE,
                    self.GRID_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                color_index = self.grid[y, x]
                pygame.draw.rect(self.screen, self.COLORS[color_index], rect)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, rect, 1)
        
        # --- Draw Flash Effects ---
        new_flashes = []
        for (pos, timer) in self.flash_effects:
            if timer > 0:
                rect = pygame.Rect(
                    self.GRID_X + pos[0] * self.CELL_SIZE,
                    self.GRID_Y + pos[1] * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE,
                )
                alpha = int(255 * (timer / 5.0))
                flash_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                flash_surface.fill((*self.COLOR_FLASH, alpha))
                self.screen.blit(flash_surface, rect.topleft)
                new_flashes.append((pos, timer - 1))
        self.flash_effects = new_flashes

        # --- Draw Cursor ---
        cursor_rect = pygame.Rect(
            self.GRID_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE,
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

    def _render_ui(self):
        # --- Right Panel for Targets ---
        panel_x = self.GRID_X + self.GRID_WIDTH + 40
        panel_y = self.GRID_Y
        
        target_text = self.font_medium.render("TARGETS", True, self.COLOR_TEXT)
        self.screen.blit(target_text, (panel_x, panel_y - 40))
        
        target_cell_size = 10
        for i in range(self.NUM_PATTERNS):
            is_completed = i < self.current_level
            is_active = i == self.current_level and not self.game_over
            
            base_y = panel_y + i * (self.GRID_SIZE * target_cell_size + 15)
            
            border_size = self.GRID_SIZE * target_cell_size + 4
            border_rect = pygame.Rect(panel_x - 3, base_y - 3, border_size + 2, border_size + 2)
            if is_completed:
                pygame.draw.rect(self.screen, self.COLORS[2], border_rect, 2) # Green for completed
            elif is_active:
                pygame.draw.rect(self.screen, self.COLOR_CURSOR, border_rect, 2) # Yellow for active
            
            pattern = self.target_patterns[i]
            for y in range(self.GRID_SIZE):
                for x in range(self.GRID_SIZE):
                    rect = pygame.Rect(
                        panel_x + x * target_cell_size,
                        base_y + y * target_cell_size,
                        target_cell_size,
                        target_cell_size,
                    )
                    color_index = pattern[y, x]
                    pygame.draw.rect(self.screen, self.COLORS[color_index], rect)

        # --- Top UI: Score and Moves ---
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        moves_text = self.font_large.render(f"MOVES: {self.moves_left}", True, self.COLOR_TEXT)
        moves_rect = moves_text.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(moves_text, moves_rect)

        # --- Game Over Message ---
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.current_level >= self.NUM_PATTERNS:
                msg = "YOU WIN!"
                color = self.COLORS[2]
            else:
                msg = "GAME OVER"
                color = self.COLORS[1]
                
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            final_score_text = self.font_medium.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            final_score_rect = final_score_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2 + 30))
            self.screen.blit(final_score_text, final_score_rect)

    def close(self):
        pygame.font.quit()
        pygame.quit()

# This allows the file to be run directly for testing and human play.
if __name__ == "__main__":
    env = GameEnv()
    
    # --- Validation ---
    def validate_implementation(self):
        print("Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        obs, info = self.reset()
        assert obs.shape == self.observation_space.shape
        assert obs.dtype == self.observation_space.dtype
        assert isinstance(info, dict)
        test_obs = self._get_observation()
        assert test_obs.shape == self.observation_space.shape
        assert test_obs.dtype == self.observation_space.dtype
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == self.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")
    validate_implementation(env)

    # --- Human Play Loop ---
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Pattern Painter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    print("\n" + "="*30)
    print("Pattern Painter - Human Play Test")
    print(env.user_guide)
    print("Press R to reset, Q to quit.")
    print("="*30 + "\n")

    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        # Event handling for quitting and resetting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    done = True
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    print("--- Environment Reset ---")
                # Since auto_advance is False, we only need to capture the paint action once.
                if event.key == pygame.K_SPACE:
                    action[1] = 1

        # Continuous key presses for movement
        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        action[0] = movement
        
        # Step the environment with the constructed action
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")
        
        if terminated:
            print("--- Episode Terminated ---")
            print(f"Final Info: {info}")
            
            # Display final screen before reset
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

            pygame.time.wait(3000)
            obs, info = env.reset()

        # Rendering
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Control the speed of the game loop

    env.close()