
# Generated: 2025-08-28T02:38:27.016850
# Source Brief: brief_04514.md
# Brief Index: 4514

        
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
    """
    A Gymnasium environment for a word-finding puzzle game.

    The player must find 10 hidden words in a grid of letters before the 60-second
    timer runs out. The game features a clean, minimalist visual style with
    clear feedback for player actions.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to start/stop selecting a word. "
        "While selecting, use arrows to extend the selection to adjacent letters. Press Shift to submit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find all the hidden words in the grid before the 60-second timer runs out. "
        "Correct words award points, but wrong submissions have a penalty."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GRID_SIZE = 14
    CELL_SIZE = 26
    GRID_MARGIN_X = 50
    NUM_WORDS = 10
    TIME_LIMIT_SECONDS = 60
    MAX_STEPS = 1800 # 60 seconds * 30 FPS, making time the primary limit

    WORD_LIST = [
        "PYTHON", "AGENT", "REWARD", "POLICY", "ACTION", "STATE", "GRID", "WORLD",
        "LEARN", "GYM", "DEEP", "SEARCH", "PUZZLE", "SOLVE", "TIMER", "SCORE",
        "VISUAL", "KERNEL", "VECTOR", "MATRIX", "TENSOR", "FLUID", "FRAME",
        "PIXEL", "RENDER", "CODE", "DEBUG", "LOGIC", "ALGO", "MODEL", "NEURAL",
        "EPOCH", "BATCH", "TRAIN", "VALID", "TEST", "PLAY", "GAME", "STEP", "BUG"
    ]

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_GRID_LINE = (55, 65, 75)
    COLOR_LETTER = (210, 220, 230)
    COLOR_CURSOR = (255, 200, 0, 100)
    COLOR_SELECTION_FILL = (0, 150, 255, 80)
    COLOR_SELECTION_LINE = (50, 180, 255)
    COLOR_TEXT = (240, 240, 240)
    COLOR_FOUND_WORD = (120, 130, 140)
    COLOR_SUCCESS = (0, 255, 150)
    COLOR_FAIL = (255, 80, 80)
    COLOR_TIMER_WARN = (255, 150, 0)
    
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
        
        # Fonts
        self.font_letter = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_ui_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_ui_text = pygame.font.SysFont("Arial", 16)
        self.font_feedback = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Calculate grid position
        self.grid_top_left = (
            self.GRID_MARGIN_X,
            (self.SCREEN_HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2
        )
        
        # Initialize state variables
        self.grid = []
        self.target_words = []
        self.found_words = set()
        self.cursor_pos = [0, 0]
        self.is_selecting = False
        self.selection_path = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.step_reward = 0.0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.feedback_message = ""
        self.feedback_color = (0,0,0)
        self.feedback_timer = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TIME_LIMIT_SECONDS
        self.step_reward = 0.0
        
        # Set to True to prevent action on first frame from a held key
        self.prev_space_held = True
        self.prev_shift_held = True

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.is_selecting = False
        self.selection_path = []
        
        self.feedback_timer = 0
        
        self._generate_grid_and_words()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        self.step_reward = 0.0
        
        if not self.game_over:
            self._update_game_state()
            self._process_input(action)
            self._check_termination()
        
        observation = self._get_observation()
        info = self._get_info()
        
        terminated = self.game_over
        reward = self.step_reward
        
        # MUST return exactly this 5-tuple
        return (
            observation,
            reward,
            terminated,
            False,  # truncated always False
            info
        )
    
    def _update_game_state(self):
        self.steps += 1
        self.time_remaining -= 1 / self.FPS
        
        if self.feedback_timer > 0:
            self.feedback_timer -= 1 / self.FPS
        
    def _process_input(self, action):
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Movement ---
        dx, dy = 0, 0
        if movement == 1: dy = -1  # up
        elif movement == 2: dy = 1 # down
        elif movement == 3: dx = -1 # left
        elif movement == 4: dx = 1 # right
        
        if dx != 0 or dy != 0:
            next_x = (self.cursor_pos[0] + dx) % self.GRID_SIZE
            next_y = (self.cursor_pos[1] + dy) % self.GRID_SIZE
            next_pos = [next_x, next_y]

            if self.is_selecting:
                last_pos = self.selection_path[-1]
                is_adjacent = abs(next_pos[0] - last_pos[0]) <= 1 and abs(next_pos[1] - last_pos[1]) <= 1
                is_new = tuple(next_pos) not in {tuple(p) for p in self.selection_path}
                
                if is_adjacent and is_new:
                    self.cursor_pos = next_pos
                    self.selection_path.append(next_pos)
                else:
                    # Invalid move during selection, break the selection
                    self.is_selecting = False
                    self.selection_path = []
                    self.cursor_pos = next_pos
            else:
                self.cursor_pos = next_pos

        # --- Handle Space (Start/End Selection) ---
        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            self.is_selecting = not self.is_selecting
            if self.is_selecting:
                self.selection_path = [self.cursor_pos]
            else:
                self.selection_path = []
                
        # --- Handle Shift (Submit Word) ---
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.is_selecting and len(self.selection_path) > 1:
            self._submit_word()

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

    def _submit_word(self):
        word = "".join([self.grid[y][x] for x, y in self.selection_path])
        
        if word in self.target_words and word not in self.found_words:
            # Correct word
            self.found_words.add(word)
            self.score += 10
            self.step_reward += 10
            self._trigger_feedback("CORRECT!", self.COLOR_SUCCESS)
            # sfx: correct_word.wav
        else:
            # Incorrect or already found word
            self.score -= 1
            self.step_reward -= 1
            self._trigger_feedback("WRONG", self.COLOR_FAIL)
            # sfx: incorrect_word.wav
            
        self.is_selecting = False
        self.selection_path = []

    def _check_termination(self):
        if self.game_over:
            return

        win = len(self.found_words) == self.NUM_WORDS
        lose_time = self.time_remaining <= 0
        lose_steps = self.steps >= self.MAX_STEPS

        if win:
            self.game_over = True
            self.step_reward += 50 # Bonus for winning
            self.score += 50
            self._trigger_feedback("YOU WIN!", self.COLOR_SUCCESS, duration=3)
        elif lose_time or lose_steps:
            self.game_over = True
            self._trigger_feedback("TIME UP!", self.COLOR_FAIL, duration=3)

    def _trigger_feedback(self, message, color, duration=0.75):
        self.feedback_message = message
        self.feedback_color = color
        self.feedback_timer = duration

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

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_top_left[0], self.grid_top_left[1], 
                                self.GRID_SIZE * self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=5)

        # Draw selection path
        if self.is_selecting and len(self.selection_path) > 0:
            # Draw connecting lines
            if len(self.selection_path) > 1:
                points = [self._get_cell_center(pos[0], pos[1]) for pos in self.selection_path]
                pygame.draw.lines(self.screen, self.COLOR_SELECTION_LINE, False, points, 5)
            # Draw circles on selected cells
            for x, y in self.selection_path:
                center = self._get_cell_center(x, y)
                pygame.gfxdraw.filled_circle(self.screen, center[0], center[1], self.CELL_SIZE // 2 - 2, self.COLOR_SELECTION_FILL)
                pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.CELL_SIZE // 2 - 2, self.COLOR_SELECTION_LINE)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_top_left[0] + self.cursor_pos[0] * self.CELL_SIZE,
            self.grid_top_left[1] + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), border_radius=4)
        self.screen.blit(s, cursor_rect.topleft)

        # Draw letters and grid lines
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                # Draw letter
                letter = self.grid[y][x]
                text_surf = self.font_letter.render(letter, True, self.COLOR_LETTER)
                text_rect = text_surf.get_rect(center=self._get_cell_center(x, y))
                self.screen.blit(text_surf, text_rect)
        
        # Draw grid lines on top for a clean look
        for i in range(self.GRID_SIZE + 1):
            start_x, end_x = self.grid_top_left[0], self.grid_top_left[0] + self.GRID_SIZE * self.CELL_SIZE
            y = self.grid_top_left[1] + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (start_x, y), (end_x, y))
            
            start_y, end_y = self.grid_top_left[1], self.grid_top_left[1] + self.GRID_SIZE * self.CELL_SIZE
            x = self.grid_top_left[0] + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, start_y), (x, end_y))

    def _render_ui(self):
        ui_x = self.grid_top_left[0] + self.GRID_SIZE * self.CELL_SIZE + 30
        
        # --- Timer ---
        timer_color = self.COLOR_TEXT
        if self.time_remaining < 10 and not self.game_over:
            # Pulse color when time is low
            pulse = (math.sin(self.steps * 0.5) + 1) / 2
            timer_color = self.COLOR_TEXT.lerp(self.COLOR_TIMER_WARN, pulse)
        
        time_text = f"{max(0, self.time_remaining):.1f}"
        timer_surf = self.font_ui_title.render(f"TIME: {time_text}", True, timer_color)
        self.screen.blit(timer_surf, (ui_x, self.grid_top_left[1]))
        
        # --- Score ---
        score_surf = self.font_ui_title.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (ui_x, self.grid_top_left[1] + 30))
        
        # --- Word List ---
        list_y = self.grid_top_left[1] + 70
        title_surf = self.font_ui_title.render("WORDS", True, self.COLOR_TEXT)
        self.screen.blit(title_surf, (ui_x, list_y))
        
        for i, word in enumerate(self.target_words):
            y_pos = list_y + 30 + i * 20
            if word in self.found_words:
                color = self.COLOR_FOUND_WORD
                word_surf = self.font_ui_text.render(word, True, color)
                self.screen.blit(word_surf, (ui_x, y_pos))
                line_y = y_pos + word_surf.get_height() // 2
                pygame.draw.line(self.screen, self.COLOR_FAIL, (ui_x, line_y), (ui_x + word_surf.get_width(), line_y), 2)
            else:
                color = self.COLOR_TEXT
                word_surf = self.font_ui_text.render(word, True, color)
                self.screen.blit(word_surf, (ui_x, y_pos))
        
        # --- Current Selection ---
        current_word = "".join([self.grid[y][x] for x, y in self.selection_path]) if self.is_selecting else ""
        sel_surf = self.font_ui_text.render(f"Selected: {current_word}", True, self.COLOR_SELECTION_LINE)
        self.screen.blit(sel_surf, (self.GRID_MARGIN_X, self.SCREEN_HEIGHT - 30))

        # --- Feedback Message ---
        if self.feedback_timer > 0:
            alpha = min(255, int(255 * (self.feedback_timer / 0.5))) # Fade out
            feedback_surf = self.font_feedback.render(self.feedback_message, True, self.feedback_color)
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(feedback_surf, feedback_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "words_found": len(self.found_words),
        }
        
    def _get_cell_center(self, x, y):
        return (
            int(self.grid_top_left[0] + (x + 0.5) * self.CELL_SIZE),
            int(self.grid_top_left[1] + (y + 0.5) * self.CELL_SIZE)
        )
        
    def _generate_grid_and_words(self):
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.target_words = self.np_random.choice(self.WORD_LIST, self.NUM_WORDS, replace=False).tolist()
        self.found_words = set()
        
        directions = [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]
        
        for word in self.target_words:
            placed = False
            for _ in range(100): # 100 attempts to place a word
                word_to_place = word if self.np_random.random() > 0.5 else word[::-1]
                
                d_idx = self.np_random.integers(0, len(directions))
                dx, dy = directions[d_idx]
                
                x_start = self.np_random.integers(0, self.GRID_SIZE)
                y_start = self.np_random.integers(0, self.GRID_SIZE)
                
                x_end, y_end = x_start + (len(word_to_place) - 1) * dx, y_start + (len(word_to_place) - 1) * dy
                
                if not (0 <= x_end < self.GRID_SIZE and 0 <= y_end < self.GRID_SIZE):
                    continue

                can_place = True
                for i in range(len(word_to_place)):
                    x, y = x_start + i * dx, y_start + i * dy
                    if self.grid[y][x] != '' and self.grid[y][x] != word_to_place[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i in range(len(word_to_place)):
                        x, y = x_start + i * dx, y_start + i * dy
                        self.grid[y][x] = word_to_place[i]
                    placed = True
                    break
            
            if not placed:
                print(f"Warning: Could not place word '{word}'")
        
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == '':
                    self.grid[y][x] = self.np_random.choice(list(alphabet))

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
    # To play the game manually
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Finder")
    
    terminated = False
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")
            
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']}")
    pygame.quit()