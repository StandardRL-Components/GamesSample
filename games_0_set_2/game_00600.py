
# Generated: 2025-08-27T14:09:34.506687
# Source Brief: brief_00600.md
# Brief Index: 600

        
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
        "Controls: ←→ to move cursor, ↑↓ to select color. Space to place peg, Shift to submit guess."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Crack the hidden 4-color code. Place your guesses and use the feedback pegs (black for correct spot, white for correct color) to deduce the solution before you run out of turns."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
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
        self.font_main = pygame.font.SysFont("Arial", 24)
        self.font_small = pygame.font.SysFont("Arial", 16)
        self.font_title = pygame.font.SysFont("Arial", 36, bold=True)

        # Game constants
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.BOARD_ROWS, self.BOARD_COLS = 8, 4
        self.NUM_COLORS = 4
        
        # Colors
        self.COLOR_BG = (35, 40, 50)
        self.COLOR_BOARD = (55, 60, 70)
        self.COLOR_HIGHLIGHT = (80, 85, 95)
        self.COLOR_TEXT = (220, 220, 230)
        self.COLOR_CURSOR = (255, 190, 0)

        self.PEG_COLORS = [
            (239, 71, 111),   # Pink
            (255, 209, 102),  # Yellow
            (6, 214, 160),    # Green
            (17, 138, 178),   # Blue
        ]
        self.EMPTY_PEG_COLOR = (45, 50, 60)
        self.FEEDBACK_CORRECT_POS = (240, 240, 240) # White
        self.FEEDBACK_CORRECT_COLOR = (20, 20, 20)  # Black

        # State variables (to be initialized in reset)
        self.secret_code = None
        self.guesses = None
        self.feedback = None
        self.current_row = None
        self.cursor_x = None
        self.selected_color_idx = None
        self.game_over = None
        self.win_state = None
        self.score = None
        self.steps = None
        self.prev_action = None
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate secret code
        self.secret_code = self.np_random.integers(0, self.NUM_COLORS, size=self.BOARD_COLS).tolist()
        
        # Initialize board state
        self.guesses = [[-1] * self.BOARD_COLS for _ in range(self.BOARD_ROWS)] # -1 for empty
        self.feedback = [(-1, -1) for _ in range(self.BOARD_ROWS)] # (black, white) pegs
        
        # Initialize player/control state
        self.current_row = 0
        self.cursor_x = 0
        self.selected_color_idx = 0
        
        # Initialize game flow state
        self.game_over = False
        self.win_state = None
        self.score = 0
        self.steps = 0
        self.prev_action = self.action_space.sample()
        self.prev_action.fill(0) # Start with no action held

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        prev_movement = self.prev_action[0]
        prev_space_held = self.prev_action[1] == 1
        prev_shift_held = self.prev_action[2] == 1

        # Detect presses (transition from 0 to 1)
        up_pressed = movement == 1 and prev_movement != 1
        down_pressed = movement == 2 and prev_movement != 2
        left_pressed = movement == 3 and prev_movement != 3
        right_pressed = movement == 4 and prev_movement != 4
        space_pressed = space_held and not prev_space_held
        shift_pressed = shift_held and not prev_shift_held

        self.steps += 1
        reward = 0

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        # --- Handle Input ---
        if left_pressed:
            self.cursor_x = (self.cursor_x - 1 + self.BOARD_COLS) % self.BOARD_COLS
        if right_pressed:
            self.cursor_x = (self.cursor_x + 1) % self.BOARD_COLS
        if up_pressed:
            self.selected_color_idx = (self.selected_color_idx - 1 + self.NUM_COLORS) % self.NUM_COLORS
        if down_pressed:
            self.selected_color_idx = (self.selected_color_idx + 1) % self.NUM_COLORS

        if space_pressed:
            # Place selected color at cursor
            self.guesses[self.current_row][self.cursor_x] = self.selected_color_idx
            # sound: "peg_place.wav"

        if shift_pressed:
            # Check if row is full before submitting
            if all(peg != -1 for peg in self.guesses[self.current_row]):
                reward = self._evaluate_guess()
                self.score += reward
                if self.win_state is not None:
                    self.game_over = True
                # sound: "guess_submit.wav"
            else:
                # sound: "error.wav"
                pass # Ignore submission of incomplete row

        self.prev_action = action
        terminated = self.game_over
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _evaluate_guess(self):
        guess = self.guesses[self.current_row]
        
        black_pegs = 0
        white_pegs = 0
        
        secret_copy = list(self.secret_code)
        guess_copy = list(guess)
        
        # First pass for black pegs (correct color and position)
        for i in range(self.BOARD_COLS):
            if guess_copy[i] == secret_copy[i]:
                black_pegs += 1
                secret_copy[i] = -1 # Mark as used to prevent re-matching
                guess_copy[i] = -2  # Mark as used
                
        # Second pass for white pegs (correct color, wrong position)
        for i in range(self.BOARD_COLS):
            if guess_copy[i] >= 0: # If this guess peg hasn't been matched yet
                try:
                    match_idx = secret_copy.index(guess_copy[i])
                    white_pegs += 1
                    secret_copy[match_idx] = -1 # Mark as used
                except ValueError:
                    pass # No match found in remaining secret code
        
        self.feedback[self.current_row] = (black_pegs, white_pegs)
        
        # Check win/loss conditions and return appropriate reward
        if black_pegs == self.BOARD_COLS:
            self.win_state = 'win'
            # sound: "win.wav"
            return 100.0
        
        self.current_row += 1
        
        if self.current_row >= self.BOARD_ROWS:
            self.win_state = 'loss'
            # sound: "loss.wav"
            return -50.0
            
        # Reward for a non-terminal guess: proportional to correctness, with a small penalty to encourage speed
        return (black_pegs * 1.0) + (white_pegs * 0.5) - 2.0
    
    def _render_peg(self, surface, color, center_x, center_y, radius):
        pygame.gfxdraw.aacircle(surface, center_x, center_y, radius, color)
        pygame.gfxdraw.filled_circle(surface, center_x, center_y, radius, color)

    def _render_game(self):
        # Board layout
        peg_radius = 20
        peg_spacing = 50
        board_width = self.BOARD_COLS * peg_spacing
        board_height = self.BOARD_ROWS * peg_spacing
        board_x = (self.SCREEN_WIDTH - board_width - 100) / 2
        board_y = (self.SCREEN_HEIGHT - board_height) / 2

        # Draw board background
        board_rect = pygame.Rect(board_x - 20, board_y - 20, board_width + 40, board_height + 40)
        pygame.draw.rect(self.screen, self.COLOR_BOARD, board_rect, border_radius=15)
        
        # Highlight active row
        if not self.game_over:
            highlight_y = board_y + self.current_row * peg_spacing - peg_spacing / 2
            highlight_rect = pygame.Rect(board_x - 10, highlight_y, board_width + 20, peg_spacing)
            pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, highlight_rect, border_radius=10)

        # Draw guess pegs and empty slots
        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
                center_x = int(board_x + c * peg_spacing + peg_radius / 2)
                center_y = int(board_y + r * peg_spacing + peg_radius / 2)
                peg_val = self.guesses[r][c]
                color = self.PEG_COLORS[peg_val] if peg_val != -1 else self.EMPTY_PEG_COLOR
                self._render_peg(self.screen, color, center_x, center_y, peg_radius - 5)

        # Draw feedback pegs
        feedback_x_start = board_x + board_width + 20
        feedback_peg_radius = 6
        for r in range(self.BOARD_ROWS):
            black, white = self.feedback[r]
            if black == -1: continue # No feedback yet for this row
            
            peg_idx = 0
            for _ in range(black):
                col = peg_idx % 2
                row = peg_idx // 2
                center_x = int(feedback_x_start + col * (feedback_peg_radius * 2.5))
                center_y = int(board_y + r * peg_spacing + row * (feedback_peg_radius * 2.5))
                self._render_peg(self.screen, self.FEEDBACK_CORRECT_POS, center_x, center_y, feedback_peg_radius)
                peg_idx += 1
            for _ in range(white):
                col = peg_idx % 2
                row = peg_idx // 2
                center_x = int(feedback_x_start + col * (feedback_peg_radius * 2.5))
                center_y = int(board_y + r * peg_spacing + row * (feedback_peg_radius * 2.5))
                self._render_peg(self.screen, self.FEEDBACK_CORRECT_COLOR, center_x, center_y, feedback_peg_radius)
                peg_idx += 1

        # Draw cursor
        if not self.game_over:
            cursor_x_pos = int(board_x + self.cursor_x * peg_spacing + peg_radius / 2)
            cursor_y_pos = int(board_y + self.current_row * peg_spacing + peg_radius / 2)
            pygame.draw.circle(self.screen, self.COLOR_CURSOR, (cursor_x_pos, cursor_y_pos), peg_radius, 3)

        # Draw color palette
        palette_y = self.SCREEN_HEIGHT - 50
        for i in range(self.NUM_COLORS):
            center_x = int(board_x + i * peg_spacing + peg_radius / 2)
            self._render_peg(self.screen, self.PEG_COLORS[i], center_x, palette_y, peg_radius - 2)
            if i == self.selected_color_idx and not self.game_over:
                pygame.draw.circle(self.screen, self.COLOR_CURSOR, (center_x, palette_y), peg_radius + 2, 3)

    def _render_ui(self):
        # Draw score and turn info
        score_text = self.font_main.render(f"Score: {self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        guesses_left = self.BOARD_ROWS - self.current_row
        turn_text = self.font_main.render(f"Guesses Left: {guesses_left}", True, self.COLOR_TEXT)
        self.screen.blit(turn_text, (self.SCREEN_WIDTH - turn_text.get_width() - 20, 20))

        # Draw game over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.win_state == 'win':
                msg_text = "YOU WIN!"
                msg_color = (100, 255, 100)
            else:
                msg_text = "YOU LOSE"
                msg_color = (255, 100, 100)
            
            msg_render = self.font_title.render(msg_text, True, msg_color)
            msg_rect = msg_render.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 50))
            self.screen.blit(msg_render, msg_rect)

            # Show the secret code
            code_text = self.font_small.render("The code was:", True, self.COLOR_TEXT)
            code_rect = code_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 10))
            self.screen.blit(code_text, code_rect)

            peg_radius = 15
            for i, color_idx in enumerate(self.secret_code):
                center_x = int(self.SCREEN_WIDTH / 2 - (1.5 * peg_radius * 2) + i * (peg_radius * 2.5))
                center_y = int(self.SCREEN_HEIGHT / 2 + 50)
                self._render_peg(self.screen, self.PEG_COLORS[color_idx], center_x, center_y, peg_radius)

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
            "win_state": self.win_state,
            "guesses_left": self.BOARD_ROWS - self.current_row
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to run the file directly to test the environment
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # This code is for human testing and is not part of the Gymnasium environment
    pygame.display.set_caption("Codebreaker Environment")
    screen = pygame.display.set_mode((640, 400))
    
    obs, info = env.reset()
    done = False
    
    action = np.array([0, 0, 0]) # [movement, space, shift]

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        # --- Keyboard to MultiDiscrete Action Mapping ---
        keys = pygame.key.get_pressed()
        
        # Movement (mutually exclusive)
        action[0] = 0 # No-op
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
            
        # Buttons (can be simultaneous)
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit frame rate for human play

    print(f"Game Over. Final Info: {info}")
    
    # Keep the window open for a few seconds to see the result
    pygame.time.wait(3000)
    
    env.close()