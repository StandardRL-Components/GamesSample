
# Generated: 2025-08-28T00:22:45.281227
# Source Brief: brief_03772.md
# Brief Index: 3772

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import Counter
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to select a number/button. Press space to input it. "
        "Press shift to delete the last digit. Select 'SUB' and press space to submit your code."
    )

    game_description = (
        "Crack the 4-digit secret code before you run out of time or attempts! After each guess, "
        "you'll get hints: green for a correct digit in the right spot, and yellow for a correct digit "
        "in the wrong spot."
    )

    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    CODE_LENGTH = 4
    MAX_ATTEMPTS = 3
    TIME_LIMIT_SECONDS = 120
    FPS = 30
    
    # --- Colors ---
    COLOR_BG = (44, 62, 80) # #2c3e50
    COLOR_SAFE = (52, 73, 94) # #34495e
    COLOR_SAFE_TRIM = (127, 140, 141) # #7f8c8d
    COLOR_TEXT = (236, 240, 241) # #ecf0f1
    COLOR_SELECTOR = (52, 152, 219) # #3498db
    COLOR_INPUT_BG = (26, 36, 46)
    COLOR_SUCCESS = (46, 204, 113) # #2ecc71
    COLOR_PARTIAL = (241, 196, 15) # #f1c400
    COLOR_FAIL = (231, 76, 60) # #e74c3c
    COLOR_DISABLED = (149, 165, 166) # #95a5a6

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.font_large = pygame.font.Font(None, 64)
        self.font_medium = pygame.font.Font(None, 40)
        self.font_small = pygame.font.Font(None, 24)
        
        self.keypad_layout = [
            ['7', '8', '9'],
            ['4', '5', '6'],
            ['1', '2', '3'],
            ['', '0', 'SUB']
        ]
        self.selector_pos = [1, 3] # Start on '0'
        
        # State variables will be initialized in reset()
        self.secret_code = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        self.time_remaining = 0
        self.attempts_remaining = 0
        self.current_input = []
        self.history = []
        self.particles = []

        # Action handling for rising edge detection
        self.last_action = np.array([0, 0, 0])
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_state = False
        
        # Generate a new secret code (unique digits)
        digits = list(range(10))
        self.np_random.shuffle(digits)
        self.secret_code = [str(d) for d in digits[:self.CODE_LENGTH]]

        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.attempts_remaining = self.MAX_ATTEMPTS
        
        self.current_input = []
        self.history = []
        self.particles = []
        self.selector_pos = [1, 3]

        self.last_action = np.array([0, 0, 0])
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.steps += 1
        
        # --- Time and Step Limit ---
        if not self.game_over:
            self.time_remaining -= 1 / self.FPS
        
        # --- Termination Conditions ---
        if self.time_remaining <= 0:
            self.game_over = True
            self.win_state = False
            self.time_remaining = 0
        if self.attempts_remaining <= 0 and not self.game_over:
            self.game_over = True
            self.win_state = False
        # Max episode length is implicitly handled by the timer
        
        # --- Action Processing (Rising Edge Detection) ---
        movement, space_press, shift_press = self._process_actions(action)
        
        if not self.game_over:
            # 1. Handle Movement
            self._handle_movement(movement)
            
            # 2. Handle Shift (Delete)
            if shift_press and self.current_input:
                self.current_input.pop()
                # sound_effect: "delete_digit.wav"

            # 3. Handle Space (Input/Submit)
            if space_press:
                button = self.keypad_layout[self.selector_pos[1]][self.selector_pos[0]]
                if button.isdigit() and len(self.current_input) < self.CODE_LENGTH:
                    self.current_input.append(button)
                    # sound_effect: "press_digit.wav"
                elif button == 'SUB' and len(self.current_input) == self.CODE_LENGTH:
                    reward = self._process_submission()
                    # sound_effect: "submit_code.wav"

        self.last_action = action
        
        terminated = self.game_over
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _process_actions(self, action):
        # Detect rising edges for event-based actions
        movement_action = action[0]
        space_action = action[1]
        shift_action = action[2]

        # Movement is continuous while held
        movement_intent = movement_action if movement_action != self.last_action[0] else 0
        
        space_press = space_action == 1 and self.last_action[1] == 0
        shift_press = shift_action == 1 and self.last_action[2] == 0

        return movement_intent, space_press, shift_press

    def _handle_movement(self, movement):
        if movement == 0: return
        
        # sound_effect: "cursor_move.wav"
        x, y = self.selector_pos
        if movement == 1: # Up
            y = (y - 1) % 4
        elif movement == 2: # Down
            y = (y + 1) % 4
        elif movement == 3: # Left
            x = (x - 1) % 3
        elif movement == 4: # Right
            x = (x + 1) % 3
        
        # Skip empty button spot
        if self.keypad_layout[y][x] == '':
            if movement == 3: # came from right
                x = (x - 1) % 3
            elif movement == 4: # came from left
                x = (x + 1) % 3

        self.selector_pos = [x, y]

    def _process_submission(self):
        # --- Feedback Calculation (Mastermind Logic) ---
        correct_pos = 0
        correct_digit = 0
        
        guess = self.current_input[:]
        secret = self.secret_code[:]
        
        # First pass for correct position
        for i in range(self.CODE_LENGTH - 1, -1, -1):
            if guess[i] == secret[i]:
                correct_pos += 1
                guess.pop(i)
                secret.pop(i)
        
        # Second pass for correct digit, wrong position
        secret_counts = Counter(secret)
        for digit in guess:
            if secret_counts[digit] > 0:
                correct_digit += 1
                secret_counts[digit] -= 1
        
        incorrect_digits = self.CODE_LENGTH - correct_pos - correct_digit
        feedback = (correct_pos, correct_digit, incorrect_digits)
        self.history.append({'guess': self.current_input[:], 'feedback': feedback})
        self.current_input = []
        self.attempts_remaining -= 1
        
        # --- Reward Calculation ---
        reward = (correct_pos * 1.0) + (correct_digit * 0.5) - (incorrect_digits * 0.1)
        self.score += reward

        # --- Check for Win Condition ---
        if correct_pos == self.CODE_LENGTH:
            self.game_over = True
            self.win_state = True
            win_reward = 100
            time_bonus = self.time_remaining * 0.1
            attempts_bonus = self.attempts_remaining * 10
            total_bonus = win_reward + time_bonus + attempts_bonus
            reward += total_bonus
            self.score += total_bonus
            # sound_effect: "win.wav"
            self._create_win_particles()
        elif self.attempts_remaining <= 0:
            # sound_effect: "lose.wav"
            pass # Game over will be set in the next step() call
            
        return reward

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
            "time_remaining": self.time_remaining,
            "attempts_remaining": self.attempts_remaining,
            "secret_code": "".join(self.secret_code) if self.game_over else "????",
        }

    def _render_game(self):
        # --- Main Safe Visual ---
        safe_rect = pygame.Rect(50, 50, self.SCREEN_WIDTH - 100, self.SCREEN_HEIGHT - 100)
        pygame.draw.rect(self.screen, self.COLOR_SAFE, safe_rect, border_radius=15)
        pygame.draw.rect(self.screen, self.COLOR_SAFE_TRIM, safe_rect, 5, border_radius=15)
        
        # --- Input Display ---
        input_bg_rect = pygame.Rect(100, 70, 240, 60)
        pygame.draw.rect(self.screen, self.COLOR_INPUT_BG, input_bg_rect, border_radius=8)
        
        input_str = "".join(self.current_input)
        # Blinking cursor effect
        cursor_visible = (self.steps // (self.FPS // 2)) % 2 == 0
        display_str = input_str
        if cursor_visible and not self.game_over and len(self.current_input) < self.CODE_LENGTH:
            display_str += "_"
        
        input_text = self.font_large.render(display_str.ljust(self.CODE_LENGTH), True, self.COLOR_TEXT)
        self.screen.blit(input_text, (110, 78))

        # --- History Display ---
        for i, entry in enumerate(self.history):
            y_pos = 150 + i * 35
            guess_str = "".join(entry['guess'])
            hist_text = self.font_medium.render(guess_str, True, self.COLOR_DISABLED)
            self.screen.blit(hist_text, (110, y_pos))
            
            # Feedback dots
            fb = entry['feedback']
            dot_x = 220
            for _ in range(fb[0]): # Green
                pygame.gfxdraw.filled_circle(self.screen, dot_x, y_pos + 15, 6, self.COLOR_SUCCESS)
                pygame.gfxdraw.aacircle(self.screen, dot_x, y_pos + 15, 6, self.COLOR_SUCCESS)
                dot_x += 20
            for _ in range(fb[1]): # Yellow
                pygame.gfxdraw.filled_circle(self.screen, dot_x, y_pos + 15, 6, self.COLOR_PARTIAL)
                pygame.gfxdraw.aacircle(self.screen, dot_x, y_pos + 15, 6, self.COLOR_PARTIAL)
                dot_x += 20
        
        # --- Keypad ---
        keypad_x_start, keypad_y_start = 380, 70
        key_w, key_h = 60, 60
        key_gap = 10
        
        for r, row in enumerate(self.keypad_layout):
            for c, key in enumerate(row):
                if key == '': continue
                
                key_rect = pygame.Rect(
                    keypad_x_start + c * (key_w + key_gap),
                    keypad_y_start + r * (key_h + key_gap),
                    key_w, key_h
                )
                
                # Draw selector
                if [c, r] == self.selector_pos:
                    selector_rect = key_rect.inflate(8, 8)
                    pygame.draw.rect(self.screen, self.COLOR_SELECTOR, selector_rect, border_radius=10)

                # Draw button
                is_sub = key == 'SUB'
                btn_color = self.COLOR_SUCCESS if is_sub else self.COLOR_SAFE_TRIM
                pygame.draw.rect(self.screen, btn_color, key_rect, border_radius=8)
                
                # Draw text
                font = self.font_small if is_sub else self.font_medium
                key_text = font.render(key, True, self.COLOR_TEXT)
                text_rect = key_text.get_rect(center=key_rect.center)
                self.screen.blit(key_text, text_rect)

        # --- Particles for Win ---
        if self.win_state:
            self._update_and_draw_particles()

    def _render_ui(self):
        # --- Timer ---
        mins, secs = divmod(int(self.time_remaining), 60)
        timer_str = f"TIME: {mins:02}:{secs:02}"
        timer_color = self.COLOR_FAIL if self.time_remaining < 10 else self.COLOR_TEXT
        timer_text = self.font_medium.render(timer_str, True, timer_color)
        self.screen.blit(timer_text, (10, 10))
        
        # --- Attempts ---
        attempts_str = f"ATTEMPTS: {self.attempts_remaining}"
        attempts_text = self.font_medium.render(attempts_str, True, self.COLOR_TEXT)
        self.screen.blit(attempts_text, (self.SCREEN_WIDTH - attempts_text.get_width() - 10, 10))

        # --- Score ---
        score_str = f"SCORE: {int(self.score)}"
        score_text = self.font_small.render(score_str, True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, self.SCREEN_HEIGHT - score_text.get_height() - 5))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        if self.win_state:
            msg = "UNLOCKED"
            color = self.COLOR_SUCCESS
        else:
            msg = "ACCESS DENIED"
            color = self.COLOR_FAIL
            
        text = self.font_large.render(msg, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
        
        code_msg = f"Code was: {''.join(self.secret_code)}"
        code_text = self.font_medium.render(code_msg, True, self.COLOR_TEXT)
        code_rect = code_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 30))

        overlay.blit(text, text_rect)
        overlay.blit(code_text, code_rect)
        self.screen.blit(overlay, (0, 0))

    def _create_win_particles(self):
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        for _ in range(100):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 6)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(30, 60)
            color = random.choice([self.COLOR_SUCCESS, self.COLOR_PARTIAL, self.COLOR_TEXT])
            self.particles.append([center_x, center_y, vx, vy, life, color])

    def _update_and_draw_particles(self):
        for p in self.particles:
            p[0] += p[2] # x += vx
            p[1] += p[3] # y += vy
            p[4] -= 1    # life -= 1
            
            if p[4] > 0:
                size = int(max(0, p[4] / 10))
                pygame.draw.circle(self.screen, p[5], (int(p[0]), int(p[1])), size)
        
        self.particles = [p for p in self.particles if p[4] > 0]

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
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Code Cracker")
    
    running = True
    total_reward = 0
    
    # Map Pygame keys to the MultiDiscrete action space
    key_map = {
        pygame.K_UP:    (1, 0, 0),
        pygame.K_DOWN:  (2, 0, 0),
        pygame.K_LEFT:  (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
        pygame.K_SPACE: (0, 1, 0),
        pygame.K_LSHIFT: (0, 0, 1),
        pygame.K_RSHIFT: (0, 0, 1),
    }

    while running:
        action = np.array([0, 0, 0]) # Default no-op action
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Get keys held down this frame
        keys = pygame.key.get_pressed()
        for key, act in key_map.items():
            if keys[key]:
                action += np.array(act)

        # Ensure movement is not combined (take the first one found)
        if action[0] > 1:
            for move_key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                if keys[move_key]:
                    action[0] = key_map[move_key][0]
                    break
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Secret Code: {info['secret_code']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        env.clock.tick(GameEnv.FPS)

    env.close()