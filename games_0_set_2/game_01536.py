
# Generated: 2025-08-27T17:27:06.933920
# Source Brief: brief_01536.md
# Brief Index: 1536

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Shift to cycle through colors. Press Space to paint the selected square."
    )

    game_description = (
        "Recreate the target image on the pixel grid within the step limit. Balance speed and accuracy for the highest score."
    )

    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and layout constants
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_DIM = 10
        self.PIXEL_SIZE = 24
        self.GRID_WIDTH = self.GRID_DIM * self.PIXEL_SIZE
        
        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_BG = (40, 45, 60)
        self.COLOR_GRID_LINE = (60, 65, 80)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_ACCENT = (100, 150, 255)
        self.COLOR_CURSOR = (255, 255, 0)
        self.COLOR_PALETTE = [
            (255, 60, 60),    # Red
            (60, 255, 60),    # Green
            (60, 120, 255),   # Blue
            (255, 255, 60),   # Yellow
            (60, 255, 255),   # Cyan
            (255, 60, 255),   # Magenta
            (255, 160, 60),   # Orange
            (240, 240, 240),  # White
        ]
        self.UNPAINTED_COLOR_INDEX = -1 # Special index for unpainted squares

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_title = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables
        self.max_steps = 200 # Adjusted from 6000 for faster episodes
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = None
        self.selected_color_idx = None
        self.target_grid = None
        self.player_grid = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.completed_rows = None
        self.completed_cols = None
        self.win_message = ""
        
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = np.array([self.GRID_DIM // 2, self.GRID_DIM // 2])
        self.selected_color_idx = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        # Generate new puzzle
        self.target_grid = self.np_random.integers(0, len(self.COLOR_PALETTE), size=(self.GRID_DIM, self.GRID_DIM))
        self.player_grid = np.full((self.GRID_DIM, self.GRID_DIM), self.UNPAINTED_COLOR_INDEX, dtype=int)
        
        self.completed_rows = set()
        self.completed_cols = set()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        step_reward = 0
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_DIM - 1)

        # 2. Handle color selection (on rising edge of shift)
        if shift_held and not self.prev_shift_held:
            self.selected_color_idx = (self.selected_color_idx + 1) % len(self.COLOR_PALETTE)
            # sfx: color_cycle

        # 3. Handle painting (on rising edge of space)
        if space_held and not self.prev_space_held:
            x, y = self.cursor_pos
            # Only paint if the color is different
            if self.player_grid[y, x] != self.selected_color_idx:
                step_reward += self._calculate_paint_reward(x, y)
                self.player_grid[y, x] = self.selected_color_idx
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # 4. Check for termination and get terminal reward
        terminated, terminal_reward = self._check_termination()
        
        total_reward = step_reward + terminal_reward
        self.score += total_reward
        
        return self._get_observation(), total_reward, terminated, False, self._get_info()

    def _calculate_paint_reward(self, x, y):
        reward = 0
        
        # Check if the placed pixel is correct
        is_correct = self.selected_color_idx == self.target_grid[y, x]
        if is_correct:
            reward += 1
            # sfx: paint_correct
        else:
            reward -= 1
            # sfx: paint_wrong

        # Temporarily apply the new color to check for row/col completion
        original_color = self.player_grid[y, x]
        self.player_grid[y, x] = self.selected_color_idx

        # Check for row completion bonus
        if y not in self.completed_rows:
            if np.array_equal(self.player_grid[y, :], self.target_grid[y, :]):
                reward += 10
                self.completed_rows.add(y)
                # sfx: row_complete

        # Check for column completion bonus
        if x not in self.completed_cols:
            if np.array_equal(self.player_grid[:, x], self.target_grid[:, x]):
                reward += 10
                self.completed_cols.add(x)
                # sfx: col_complete

        # Revert the temporary change
        self.player_grid[y, x] = original_color
        
        return reward

    def _check_termination(self):
        terminated = False
        terminal_reward = 0
        
        # Win condition: 100% accuracy
        if np.array_equal(self.player_grid, self.target_grid):
            terminated = True
            terminal_reward = 100
            self.win_message = "PERFECT!"
            # sfx: win_fanfare
        
        # Loss condition: Ran out of steps
        elif self.steps >= self.max_steps:
            terminated = True
            terminal_reward = -50
            self.win_message = "TIME'S UP!"
            # sfx: lose_buzzer

        if terminated:
            self.game_over = True
            
        return terminated, terminal_reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Calculate layout positions
        grid_y = 90
        total_width = self.GRID_WIDTH * 2 + 40
        player_grid_x = (self.SCREEN_WIDTH - total_width) // 2
        target_grid_x = player_grid_x + self.GRID_WIDTH + 40

        # Draw Player Grid
        self._draw_grid(self.player_grid, player_grid_x, grid_y, "YOUR GRID")
        
        # Draw Target Grid
        self._draw_grid(self.target_grid, target_grid_x, grid_y, "TARGET")

        # Draw Cursor on Player Grid
        cursor_x, cursor_y = self.cursor_pos
        rect = (
            player_grid_x + cursor_x * self.PIXEL_SIZE,
            grid_y + cursor_y * self.PIXEL_SIZE,
            self.PIXEL_SIZE, self.PIXEL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 3)

        # Draw Color Palette
        palette_y = grid_y + self.GRID_WIDTH + 20
        palette_width = len(self.COLOR_PALETTE) * (self.PIXEL_SIZE + 5) - 5
        palette_x = (self.SCREEN_WIDTH - palette_width) // 2
        
        for i, color in enumerate(self.COLOR_PALETTE):
            rect = (palette_x + i * (self.PIXEL_SIZE + 5), palette_y, self.PIXEL_SIZE, self.PIXEL_SIZE)
            pygame.draw.rect(self.screen, color, rect)
            if i == self.selected_color_idx:
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 3)

    def _draw_grid(self, grid_data, x_offset, y_offset, title):
        # Draw background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (x_offset, y_offset, self.GRID_WIDTH, self.GRID_WIDTH))
        # Draw title
        title_surf = self.font_title.render(title, True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(title_surf, (x_offset + (self.GRID_WIDTH - title_surf.get_width())//2, y_offset - 30))

        # Draw pixels
        for y in range(self.GRID_DIM):
            for x in range(self.GRID_DIM):
                color_idx = grid_data[y, x]
                if color_idx != self.UNPAINTED_COLOR_INDEX:
                    color = self.COLOR_PALETTE[color_idx]
                    rect = (x_offset + x * self.PIXEL_SIZE, y_offset + y * self.PIXEL_SIZE, self.PIXEL_SIZE, self.PIXEL_SIZE)
                    pygame.draw.rect(self.screen, color, rect)
        
        # Draw grid lines
        for i in range(self.GRID_DIM + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x_offset, y_offset + i * self.PIXEL_SIZE), (x_offset + self.GRID_WIDTH, y_offset + i * self.PIXEL_SIZE))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x_offset + i * self.PIXEL_SIZE, y_offset), (x_offset + i * self.PIXEL_SIZE, y_offset + self.GRID_WIDTH))

    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (20, 20))

        # Steps Left
        steps_left = max(0, self.max_steps - self.steps)
        steps_text = f"STEPS LEFT: {steps_left}"
        steps_surf = self.font_ui.render(steps_text, True, self.COLOR_TEXT)
        self.screen.blit(steps_surf, (self.SCREEN_WIDTH - steps_surf.get_width() - 20, 20))

        # Accuracy
        accuracy = np.mean(self.player_grid == self.target_grid) * 100
        acc_text = f"ACCURACY: {accuracy:.1f}%"
        acc_surf = self.font_ui.render(acc_text, True, self.COLOR_TEXT)
        self.screen.blit(acc_surf, (20, 40))

        # Game Over Overlay
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg_surf = self.font_game_over.render(self.win_message, True, (255, 255, 255))
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "accuracy": np.mean(self.player_grid == self.target_grid)
        }
    
    def close(self):
        pygame.font.quit()
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

if __name__ == "__main__":
    # This block allows you to play the game with keyboard controls
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Mapping from Pygame keys to action space components
    key_to_action = {
        pygame.K_UP:    (1, 0, 0),
        pygame.K_DOWN:  (2, 0, 0),
        pygame.K_LEFT:  (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
        pygame.K_SPACE: (0, 1, 0),
        pygame.K_LSHIFT: (0, 0, 1),
        pygame.K_RSHIFT: (0, 0, 1),
    }

    # Pygame setup for human play
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Painter")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get keys held down this frame
        keys = pygame.key.get_pressed()
        
        # Combine actions - allows moving while holding space/shift
        for key, act in key_to_action.items():
            if keys[key]:
                action[0] = act[0] if act[0] != 0 else action[0]
                action[1] = act[1] if act[1] != 0 else action[1]
                action[2] = act[2] if act[2] != 0 else action[2]

        if not done:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if reward != 0:
                print(f"Step: {info['steps']}, Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        if done:
            # Allow resetting the game by pressing 'R'
            if keys[pygame.K_r]:
                print("--- RESETTING GAME ---")
                obs, info = env.reset()
                done = False

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for human play

    env.close()