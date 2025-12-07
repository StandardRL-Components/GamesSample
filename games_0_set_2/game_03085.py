
# Generated: 2025-08-28T06:56:36.989248
# Source Brief: brief_03085.md
# Brief Index: 3085

        
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
        "Controls: Arrow keys to move cursor. Hold Space to select a word. Press Shift to submit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find all the hidden words in the grid before time runs out. Select words by holding Space and moving the cursor from the start to the end letter, then press Shift to submit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    NUM_WORDS = 20
    TIME_LIMIT_SECONDS = 60
    FPS = 30

    # Colors
    COLOR_BG = (28, 36, 48)
    COLOR_GRID_LINES = (48, 56, 68)
    COLOR_LETTER = (220, 220, 220)
    COLOR_UI_TEXT = (200, 200, 210)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION = (255, 200, 0, 100)  # Yellow, semi-transparent
    COLOR_FOUND = (0, 255, 100, 80)     # Green, semi-transparent
    COLOR_FEEDBACK_SUCCESS = (0, 255, 100, 50)
    COLOR_FEEDBACK_FAIL = (255, 50, 50, 50)

    WORD_LIST = [
        "PYTHON", "GYMNASIUM", "REWARD", "ACTION", "AGENT", "POLICY", "STATE",
        "EPISODE", "LEARNING", "NEURAL", "NETWORK", "TENSOR", "FLASK", "STREAMLIT",
        "DOCKER", "CLOUD", "REACT", "ANGULAR", "VUE", "JAVASCRIPT", "HTML", "CSS",
        "ALGORITHM", "SEARCH", "PUZZLE", "GRID", "TIMER", "SCORE", "VECTOR",
        "MATRIX", "DEEP", "MODEL", "TRAIN", "TEST", "VALIDATE", "KERAS", "PYTORCH"
    ]

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

        self.font_letter = pygame.font.Font(None, 18)
        self.font_ui_large = pygame.font.Font(None, 32)
        self.font_ui_medium = pygame.font.Font(None, 24)
        self.font_ui_small = pygame.font.Font(None, 16)
        
        self.grid_area_width = self.SCREEN_HEIGHT - 40
        self.cell_size = self.grid_area_width // self.GRID_SIZE
        self.grid_start_x = 20
        self.grid_start_y = (self.SCREEN_HEIGHT - self.grid_area_width) // 2

        self.max_steps = self.TIME_LIMIT_SECONDS * self.FPS
        
        # State variables will be initialized in reset()
        self.grid = []
        self.words_to_find = []
        self.word_locations = {}
        self.found_words_info = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = (0, 0)
        self.selection_start = None
        self.current_selection_path = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.feedback_effect = None # (color, duration)
        
        # This is here to ensure all attributes are defined before validation
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_puzzle()

        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.selection_start = None
        self.current_selection_path = []
        self.found_words_info = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.feedback_effect = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        # 1. Handle Movement
        if movement == 1: self.cursor_pos = (self.cursor_pos[0], max(0, self.cursor_pos[1] - 1))
        elif movement == 2: self.cursor_pos = (self.cursor_pos[0], min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1))
        elif movement == 3: self.cursor_pos = (max(0, self.cursor_pos[0] - 1), self.cursor_pos[1])
        elif movement == 4: self.cursor_pos = (min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1), self.cursor_pos[1])

        # 2. Handle Selection
        if space_held and not self.prev_space_held:
            # Start a new selection
            self.selection_start = self.cursor_pos
            self.current_selection_path = [self.cursor_pos]
        elif space_held and self.selection_start is not None:
            # Update selection path while dragging
            self.current_selection_path = self._get_line_path(self.selection_start, self.cursor_pos)
        
        # 3. Handle Submission (on rising edge of shift)
        if shift_held and not self.prev_shift_held and self.selection_start is not None and len(self.current_selection_path) > 1:
            selected_word = "".join([self.grid[y][x] for x, y in self.current_selection_path])
            
            # Check if word or its reverse is in the list of words to find
            found = False
            for word in self.words_to_find:
                if word not in [info['word'] for info in self.found_words_info]:
                    if selected_word == word or selected_word[::-1] == word:
                        # Correct word found!
                        reward += 10
                        self.score += 10
                        self.found_words_info.append({
                            "word": word,
                            "path": self.current_selection_path
                        })
                        found = True
                        self.feedback_effect = (self.COLOR_FEEDBACK_SUCCESS, 5) # sound: success_ding.wav
                        break
            
            if not found:
                # Incorrect submission
                # Per brief, give continuous feedback on submission
                continuous_reward = 0
                for x, y in self.current_selection_path:
                    # Check if this letter is part of any unfound word
                    in_any_word = any(
                        (x, y) in self.word_locations[w]
                        for w in self.words_to_find
                        if w not in [info['word'] for info in self.found_words_info]
                    )
                    continuous_reward += 0.1 if in_any_word else -0.1
                reward += continuous_reward
                self.feedback_effect = (self.COLOR_FEEDBACK_FAIL, 5) # sound: fail_buzz.wav

            # Reset selection after any submission attempt
            self.selection_start = None
            self.current_selection_path = []

        # Update previous button states
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # Update step counter and time
        self.steps += 1
        
        # Check for termination conditions
        terminated = False
        if len(self.found_words_info) == self.NUM_WORDS:
            reward += 50 # Victory bonus
            self.score += 50
            terminated = True
            self.game_over = True
        elif self.steps >= self.max_steps:
            reward -= 50 # Time out penalty
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_found": len(self.found_words_info),
            "time_remaining": max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS)),
        }

    def _generate_puzzle(self):
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.words_to_find = random.sample([w for w in self.WORD_LIST if 4 <= len(w) <= 8], self.NUM_WORDS)
        self.word_locations = {}

        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]

        for word in sorted(self.words_to_find, key=len, reverse=True):
            placed = False
            for _ in range(100): # 100 placement attempts per word
                random.shuffle(directions)
                direction = random.choice(directions)
                dx, dy = direction
                
                start_x = self.np_random.integers(0, self.GRID_SIZE)
                start_y = self.np_random.integers(0, self.GRID_SIZE)
                
                end_x = start_x + (len(word) - 1) * dx
                end_y = start_y + (len(word) - 1) * dy

                if not (0 <= end_x < self.GRID_SIZE and 0 <= end_y < self.GRID_SIZE):
                    continue

                can_place = True
                path = []
                for i in range(len(word)):
                    x, y = start_x + i * dx, start_y + i * dy
                    path.append((x, y))
                    if self.grid[y][x] != '' and self.grid[y][x] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i, (x, y) in enumerate(path):
                        self.grid[y][x] = word[i]
                    self.word_locations[word] = path
                    placed = True
                    break
            
            if not placed:
                # This should ideally not happen. For robustness, we regenerate the puzzle.
                return self._generate_puzzle()
        
        # Fill empty cells with random letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == '':
                    self.grid[y][x] = self.np_random.choice(list(alphabet))

    def _get_line_path(self, start_pos, end_pos):
        x1, y1 = start_pos
        x2, y2 = end_pos
        path = []
        
        dx, dy = x2 - x1, y2 - y1

        # Horizontal, Vertical, or 45-degree Diagonal
        if dx == 0 or dy == 0 or abs(dx) == abs(dy):
            steps = max(abs(dx), abs(dy))
            if steps == 0: return [(x1, y1)]
            
            x_step = dx / steps
            y_step = dy / steps
            
            for i in range(steps + 1):
                path.append((round(x1 + i * x_step), round(y1 + i * y_step)))
        else:
            # Invalid line, just return start point.
            return [start_pos]
        return path

    def _render_game(self):
        # Draw found word highlights (underneath letters)
        for info in self.found_words_info:
            if len(info['path']) > 1:
                start_pos = info['path'][0]
                end_pos = info['path'][-1]
                start_center = (self.grid_start_x + start_pos[0] * self.cell_size + self.cell_size // 2,
                                self.grid_start_y + start_pos[1] * self.cell_size + self.cell_size // 2)
                end_center = (self.grid_start_x + end_pos[0] * self.cell_size + self.cell_size // 2,
                              self.grid_start_y + end_pos[1] * self.cell_size + self.cell_size // 2)
                pygame.draw.line(self.screen, self.COLOR_FOUND[:3], start_center, end_center, self.cell_size - 4)

        # Draw current selection highlight
        if self.selection_start is not None and len(self.current_selection_path) > 1:
            start_pos = self.current_selection_path[0]
            end_pos = self.current_selection_path[-1]
            start_center = (self.grid_start_x + start_pos[0] * self.cell_size + self.cell_size // 2,
                            self.grid_start_y + start_pos[1] * self.cell_size + self.cell_size // 2)
            end_center = (self.grid_start_x + end_pos[0] * self.cell_size + self.cell_size // 2,
                          self.grid_start_y + end_pos[1] * self.cell_size + self.cell_size // 2)
            pygame.draw.line(self.screen, self.COLOR_SELECTION[:3], start_center, end_center, self.cell_size - 4)

        # Draw grid and letters
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(self.grid_start_x + x * self.cell_size,
                                   self.grid_start_y + y * self.cell_size,
                                   self.cell_size, self.cell_size)
                
                letter_surf = self.font_letter.render(self.grid[y][x], True, self.COLOR_LETTER)
                letter_rect = letter_surf.get_rect(center=rect.center)
                self.screen.blit(letter_surf, letter_rect)

        # Draw cursor
        cursor_rect = pygame.Rect(self.grid_start_x + self.cursor_pos[0] * self.cell_size,
                                  self.grid_start_y + self.cursor_pos[1] * self.cell_size,
                                  self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)

    def _render_ui(self):
        # Time
        time_left = max(0, self.TIME_LIMIT_SECONDS - (self.steps / self.FPS))
        minutes, seconds = divmod(int(time_left), 60)
        time_str = f"{minutes:02}:{seconds:02}"
        time_color = self.COLOR_UI_TEXT if time_left > 10 else (255, 80, 80)
        time_surf = self.font_ui_large.render(time_str, True, time_color)
        time_rect = time_surf.get_rect(centerx=(self.grid_start_x + self.grid_area_width) / 2, top=5)
        self.screen.blit(time_surf, time_rect)

        # Score
        score_str = f"Score: {self.score}"
        score_surf = self.font_ui_medium.render(score_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, 10))

        # Word List
        word_list_x = self.grid_start_x + self.grid_area_width + 30
        word_list_y = 40
        
        found_word_set = {info['word'] for info in self.found_words_info}
        
        for i, word in enumerate(self.words_to_find):
            is_found = word in found_word_set
            color = (100, 200, 120) if is_found else (150, 150, 160)
            word_surf = self.font_ui_small.render(word, True, color)
            word_pos = (word_list_x, word_list_y + i * 17)
            self.screen.blit(word_surf, word_pos)
            if is_found:
                pygame.draw.line(self.screen, color, word_pos, (word_pos[0] + word_surf.get_width(), word_pos[1] + word_surf.get_height()//2), 2)
        
        # Feedback effect
        if self.feedback_effect:
            color, duration = self.feedback_effect
            if duration > 0:
                overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                overlay.fill(color)
                self.screen.blit(overlay, (0, 0))
                self.feedback_effect = (color, duration - 1)
            else:
                self.feedback_effect = None
                
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
        
        # Test game-specific assertions
        assert self.TIME_LIMIT_SECONDS == 60
        assert self.NUM_WORDS == 20
        assert len(self.words_to_find) == self.NUM_WORDS
        # Check if all words are in the grid (proxy for successful generation)
        for word in self.words_to_find:
            word_path = self.word_locations.get(word)
            assert word_path is not None, f"Word '{word}' was not placed in the grid."
            reconstructed_word = "".join([self.grid[y][x] for x, y in word_path])
            assert reconstructed_word == word, f"Word '{word}' mismatch in grid."

        print("✓ Implementation validated successfully")