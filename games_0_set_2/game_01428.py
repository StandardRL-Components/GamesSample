
# Generated: 2025-08-27T17:07:22.510548
# Source Brief: brief_01428.md
# Brief Index: 1428

        
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
        "Controls: Use ←→ to move the cursor. Use ↑↓ to change the selected digit. "
        "Press Space to submit your guess. Hold Shift to clear a digit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Crack the secret 4-digit code. After each guess, you'll see how many digits are correct "
        "and in the right spot (green) and how many are correct but in the wrong spot (yellow). You have 10 tries."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_ATTEMPTS = 10
        self.CODE_LENGTH = 4
        self.MAX_STEPS = 1000

        # --- Colors ---
        self.COLOR_BG = (30, 35, 40)
        self.COLOR_GRID = (60, 65, 70)
        self.COLOR_TEXT = (220, 220, 225)
        self.COLOR_PLACEHOLDER = (80, 85, 90)
        self.COLOR_CURSOR = (70, 150, 255)
        self.COLOR_GREEN_PEG = (40, 200, 120)
        self.COLOR_YELLOW_PEG = (255, 200, 50)
        self.COLOR_OVERLAY = (30, 35, 40, 200) # RGBA

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
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)
        self.font_medium = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)

        # --- Game State (initialized in reset) ---
        self.secret_code = []
        self.guesses = []
        self.feedback = []
        self.current_attempt = 0
        self.current_guess = []
        self.cursor_pos = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.score = 0
        self.message = ""
        self.prev_space_held = False

        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate a new secret code (digits can repeat)
        self.secret_code = self.np_random.integers(0, 10, size=self.CODE_LENGTH).tolist()

        # Reset game state
        self.guesses = []
        self.feedback = []
        self.current_attempt = 0
        self.current_guess = [-1] * self.CODE_LENGTH  # -1 represents an empty slot
        self.cursor_pos = 0
        self.game_over = False
        self.win = False
        self.steps = 0
        self.score = 0.0
        self.message = ""
        self.prev_space_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        is_submit_press = space_held and not self.prev_space_held

        if is_submit_press:
            # Check if the guess is complete
            if -1 not in self.current_guess:
                # Submit guess and calculate feedback/reward
                reward = self._submit_guess()
                # sound_effect: 'submit_guess'
        elif shift_held:
            # Clear the digit at the cursor
            if self.current_guess[self.cursor_pos] != -1:
                self.current_guess[self.cursor_pos] = -1
                # sound_effect: 'clear_digit'
        elif movement != 0:
            # Handle cursor movement and digit changes
            self._handle_movement(movement)
        
        self.prev_space_held = space_held
        
        terminated = self.game_over or self.steps >= self.MAX_STEPS
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _submit_guess(self):
        # 1. Add current guess to history
        self.guesses.append(list(self.current_guess))

        # 2. Calculate feedback
        greens, yellows = self._calculate_feedback()
        self.feedback.append((greens, yellows))
        
        # 3. Calculate reward for this guess
        reward = greens * 1.0 + yellows * 0.5
        self.score += reward

        # 4. Check for win/loss
        if greens == self.CODE_LENGTH:
            self.game_over = True
            self.win = True
            self.message = "CODE CRACKED!"
            reward += 50.0  # Win bonus
            self.score += 50.0
            # sound_effect: 'win'
        elif self.current_attempt + 1 >= self.MAX_ATTEMPTS:
            self.game_over = True
            self.win = False
            self.message = "ATTEMPTS EXHAUSTED"
            # On loss, the total reward for this step is a flat penalty
            reward = -100.0
            self.score = -100.0 # Final score is the penalty
            # sound_effect: 'lose'

        # 5. Update state for next turn
        self.current_attempt += 1
        self.current_guess = [-1] * self.CODE_LENGTH
        self.cursor_pos = 0
        
        return reward

    def _calculate_feedback(self):
        guess_copy = self.current_guess.copy()
        code_copy = self.secret_code.copy()
        
        greens = 0
        yellows = 0

        # First pass for correct digit in correct position (greens)
        for i in range(self.CODE_LENGTH):
            if guess_copy[i] == code_copy[i]:
                greens += 1
                guess_copy[i] = -1  # Mark as used in guess
                code_copy[i] = -2   # Mark as used in code
        
        # Second pass for correct digit in wrong position (yellows)
        for i in range(self.CODE_LENGTH):
            if guess_copy[i] != -1: # If not already counted as green
                try:
                    # Find if the digit exists in the remaining code
                    idx = code_copy.index(guess_copy[i])
                    yellows += 1
                    code_copy[idx] = -2 # Mark as used in code
                except ValueError:
                    pass # Digit not found
        
        return greens, yellows

    def _handle_movement(self, movement):
        # sound_effect: 'cursor_move' or 'digit_change'
        if movement == 1:  # Up
            if self.current_guess[self.cursor_pos] == -1:
                self.current_guess[self.cursor_pos] = 0
            else:
                self.current_guess[self.cursor_pos] = (self.current_guess[self.cursor_pos] + 1) % 10
        elif movement == 2:  # Down
            if self.current_guess[self.cursor_pos] == -1:
                self.current_guess[self.cursor_pos] = 9
            else:
                self.current_guess[self.cursor_pos] = (self.current_guess[self.cursor_pos] - 1 + 10) % 10
        elif movement == 3:  # Left
            self.cursor_pos = max(0, self.cursor_pos - 1)
        elif movement == 4:  # Right
            self.cursor_pos = min(self.CODE_LENGTH - 1, self.cursor_pos + 1)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "attempts_left": self.MAX_ATTEMPTS - self.current_attempt,
            "win": self.win,
        }

    def _render_game(self):
        cell_size = 50
        peg_size = 8
        grid_start_x = (self.WIDTH - (self.CODE_LENGTH * cell_size + (self.CODE_LENGTH + 1) * peg_size * 2)) // 2
        
        # Draw current guess area
        current_y = 80
        for j in range(self.CODE_LENGTH):
            rect = pygame.Rect(grid_start_x + j * cell_size, current_y, cell_size, cell_size)
            
            # Pulsing cursor effect
            if j == self.cursor_pos and not self.game_over:
                alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.005)
                cursor_color = (*self.COLOR_CURSOR, alpha)
                cursor_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
                pygame.draw.rect(cursor_surface, cursor_color, cursor_surface.get_rect(), border_radius=8)
                self.screen.blit(cursor_surface, rect.topleft)

            pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2, border_radius=5)
            
            digit = self.current_guess[j]
            if digit != -1:
                text_surf = self.font_large.render(str(digit), True, self.COLOR_TEXT)
            else:
                text_surf = self.font_large.render("_", True, self.COLOR_PLACEHOLDER)
            
            text_rect = text_surf.get_rect(center=rect.center)
            self.screen.blit(text_surf, text_rect)

        # Draw guess history
        history_start_y = current_y + cell_size + 20
        for i in range(self.MAX_ATTEMPTS):
            y_pos = history_start_y + i * (cell_size * 0.7)
            
            # Dim older guesses
            alpha = 255 - (self.MAX_ATTEMPTS - 1 - i) * 15
            color = (*self.COLOR_GRID[:3], alpha)
            
            # Render guess digits
            if i < len(self.guesses):
                for j in range(self.CODE_LENGTH):
                    rect = pygame.Rect(grid_start_x + j * cell_size, y_pos, cell_size, cell_size * 0.6)
                    digit_str = str(self.guesses[i][j])
                    text_surf = self.font_medium.render(digit_str, True, self.COLOR_TEXT)
                    text_surf.set_alpha(alpha)
                    text_rect = text_surf.get_rect(center=rect.center)
                    self.screen.blit(text_surf, text_rect)
            else: # Render empty slots
                for j in range(self.CODE_LENGTH):
                    rect = pygame.Rect(grid_start_x + j * cell_size, y_pos, cell_size, cell_size * 0.6)
                    pygame.draw.rect(self.screen, color, rect, 1, border_radius=3)

            # Draw feedback pegs
            if i < len(self.feedback):
                greens, yellows = self.feedback[i]
                peg_x = grid_start_x + self.CODE_LENGTH * cell_size + peg_size + 10
                peg_y = y_pos + (cell_size * 0.6) / 2
                
                peg_colors = [self.COLOR_GREEN_PEG] * greens + [self.COLOR_YELLOW_PEG] * yellows
                for k, p_color in enumerate(peg_colors):
                    px = int(peg_x + k * (peg_size * 2.5))
                    py = int(peg_y)
                    pygame.gfxdraw.aacircle(self.screen, px, py, peg_size, p_color)
                    pygame.gfxdraw.filled_circle(self.screen, px, py, peg_size, p_color)

    def _render_ui(self):
        # Title
        title_surf = self.font_medium.render("CODEBREAKER", True, self.COLOR_TEXT)
        title_rect = title_surf.get_rect(center=(self.WIDTH // 2, 25))
        self.screen.blit(title_surf, title_rect)

        # Attempts Left
        attempts_text = f"ATTEMPTS: {self.MAX_ATTEMPTS - self.current_attempt}"
        attempts_surf = self.font_small.render(attempts_text, True, self.COLOR_TEXT)
        self.screen.blit(attempts_surf, (20, 20))

        # Score
        score_text = f"SCORE: {self.score:.1f}"
        score_surf = self.font_small.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topright=(self.WIDTH - 20, 20))
        self.screen.blit(score_surf, score_rect)

    def _render_game_over(self):
        # Semi-transparent overlay
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        # Win/Loss Message
        msg_color = self.COLOR_GREEN_PEG if self.win else self.COLOR_YELLOW_PEG
        msg_surf = self.font_large.render(self.message, True, msg_color)
        msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 - 40))
        self.screen.blit(msg_surf, msg_rect)

        # Reveal Secret Code
        code_text = "The code was: " + "".join(map(str, self.secret_code))
        code_surf = self.font_medium.render(code_text, True, self.COLOR_TEXT)
        code_rect = code_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2 + 20))
        self.screen.blit(code_surf, code_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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
    # This block allows you to run the game directly for testing
    env = GameEnv()
    env.validate_implementation()
    
    obs, info = env.reset()
    done = False
    
    print("\n--- Interactive Test ---")
    print(env.game_description)
    print(env.user_guide)
    
    # Game loop for human player
    running = True
    while running:
        # Pygame event handling
        movement, space_press, shift_press = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_SPACE:
                    space_press = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_press = 1
                elif event.key == pygame.K_r: # Reset
                    print("--- Resetting Environment ---")
                    obs, info = env.reset()
                elif event.key == pygame.K_q: # Quit
                    running = False

        # Create an action and step the environment
        action = [movement, space_press, shift_press]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if movement or space_press or shift_press:
            if reward != 0:
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}")

        if terminated and not done:
            print(f"Game Over! Final Score: {info['score']:.2f}. Press 'R' to play again or 'Q' to quit.")
            done = True # Prevent repeated printing
        
        if not terminated:
            done = False

        # Convert observation back to a Pygame surface to display
        display_surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        
        # Create a display window if one doesn't exist
        try:
            screen = pygame.display.get_surface()
            if screen is None: raise AttributeError
        except (pygame.error, AttributeError):
            screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
            pygame.display.set_caption("CodeBreaker Test")
            
        screen.blit(display_surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for the interactive loop
        
    env.close()