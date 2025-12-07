
# Generated: 2025-08-28T07:08:38.266248
# Source Brief: brief_03133.md
# Brief Index: 3133

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import string
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select letters. "
        "Press Shift to submit your selected word."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find all 10 hidden words in the grid before the 60-second timer runs out. "
        "Correct words turn green, incorrect submissions flash red."
    )

    # Frames auto-advance for the real-time timer.
    auto_advance = True

    # --- Constants ---
    # Visuals
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_LINES = (50, 60, 70)
    COLOR_LETTER = (220, 230, 240)
    COLOR_CURSOR = (255, 180, 0)
    COLOR_SELECTION_PATH = (255, 180, 0, 180)
    COLOR_FOUND_LETTER = (100, 110, 120)
    COLOR_UI_TEXT = (200, 210, 220)
    COLOR_UI_SUCCESS = (0, 255, 120)
    COLOR_UI_FAILURE = (255, 80, 80)
    COLOR_UI_FOUND_WORD = (120, 130, 140)

    # Game settings
    GRID_SIZE = (18, 11)  # Width, Height
    CELL_SIZE = 32
    NUM_WORDS = 10
    MAX_TIME = 60  # seconds
    MAX_STEPS = 1800 # 60 seconds * 30 FPS
    WORD_BANK = [
        "PYTHON", "AGENT", "REWARD", "ACTION", "POLICY", "STATE", "GRID", "GYM",
        "LEARNING", "TENSOR", "VECTOR", "DEEP", "MODEL", "NEURAL", "SEARCH", "GAME",
        "SOLVE", "PUZZLE", "CODE", "VISUAL", "PLAY", "STEP", "RESET", "FRAME", "PIXEL"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400
        self.grid_width, self.grid_height = self.GRID_SIZE
        
        self.grid_pixel_width = self.grid_width * self.CELL_SIZE
        self.grid_pixel_height = self.grid_height * self.CELL_SIZE
        
        self.grid_offset_x = 40
        self.grid_offset_y = (self.screen_height - self.grid_pixel_height) // 2

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_letter = pygame.font.SysFont("Consolas", 22, bold=True)
        self.font_ui = pygame.font.SysFont("Segoe UI", 18)
        self.font_ui_large = pygame.font.SysFont("Segoe UI", 28, bold=True)
        self.font_ui_strikethrough = pygame.font.SysFont("Segoe UI", 18)
        self.font_ui_strikethrough.set_strikethrough(True)

        # Initialize state variables
        self.grid = []
        self.words_to_find = []
        self.word_data = {}
        self.cursor_pos = [0, 0]
        self.current_selection_coords = []
        self.found_words = set()
        self.time_left = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.space_was_held = False
        self.shift_was_held = False
        
        self.feedback_flash_timer = 0
        self.feedback_flash_color = (0,0,0)

        # Initialize RNG
        self.np_random = None

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_TIME
        self.found_words = set()
        self.current_selection_coords = []
        self.space_was_held = True # Prevent action on first frame
        self.shift_was_held = True # Prevent action on first frame
        self.feedback_flash_timer = 0
        
        self.cursor_pos = [self.grid_width // 2, self.grid_height // 2]
        
        # Procedurally generate grid until a valid one is made
        generation_success = False
        while not generation_success:
            generation_success = self._generate_grid()

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        self.words_to_find = random.sample(self.WORD_BANK, self.NUM_WORDS)
        self.word_data = {}

        directions = [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]

        for word in self.words_to_find:
            placed = False
            for _ in range(100): # 100 placement attempts per word
                random.shuffle(directions)
                direction = random.choice(directions)
                dx, dy = direction
                
                start_x = self.np_random.integers(0, self.grid_width)
                start_y = self.np_random.integers(0, self.grid_height)

                end_x = start_x + (len(word) - 1) * dx
                end_y = start_y + (len(word) - 1) * dy

                if not (0 <= end_x < self.grid_width and 0 <= end_y < self.grid_height):
                    continue

                can_place = True
                coords = []
                for i in range(len(word)):
                    x, y = start_x + i * dx, start_y + i * dy
                    coords.append((x, y))
                    if self.grid[y][x] != '' and self.grid[y][x] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i, (x, y) in enumerate(coords):
                        self.grid[y][x] = word[i]
                    self.word_data[word] = {'coords': coords, 'found': False}
                    placed = True
                    break
            
            if not placed:
                return False # Generation failed, will retry

        # Fill empty cells with random letters
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.grid[y][x] == '':
                    self.grid[y][x] = random.choice(string.ascii_uppercase)
        
        return True

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        
        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Detect rising edge of buttons
        space_pressed = space_held and not self.space_was_held
        shift_pressed = shift_held and not self.shift_was_held
        self.space_was_held = space_held
        self.shift_was_held = shift_held

        # --- Update Game Logic ---
        self.steps += 1
        self.time_left -= 1.0 / 30.0 # Assuming 30 FPS
        if self.feedback_flash_timer > 0:
            self.feedback_flash_timer -= 1

        # 1. Handle Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
        elif movement == 2: self.cursor_pos[1] = min(self.grid_height - 1, self.cursor_pos[1] + 1) # Down
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
        elif movement == 4: self.cursor_pos[0] = min(self.grid_width - 1, self.cursor_pos[0] + 1) # Right
        
        # 2. Handle Space Press (Letter Selection)
        if space_pressed:
            cursor_coord = tuple(self.cursor_pos)
            if cursor_coord in self.current_selection_coords:
                # Clear selection if selecting an already selected letter
                self.current_selection_coords = []
                # sound: clear_selection.wav
            else:
                self.current_selection_coords.append(cursor_coord)
                # sound: select_letter.wav

        # 3. Handle Shift Press (Word Submission)
        if shift_pressed:
            reward += self._check_submission()
            self.current_selection_coords = []

        # --- Check Termination Conditions ---
        terminated = False
        if len(self.found_words) == self.NUM_WORDS:
            reward += 50  # Win bonus
            terminated = True
            self.game_over = True
            self._set_feedback_flash(self.COLOR_UI_SUCCESS, 30)
        elif self.time_left <= 0:
            reward -= 50  # Time out penalty
            terminated = True
            self.game_over = True
            self._set_feedback_flash(self.COLOR_UI_FAILURE, 30)
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
        
        self.score += reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_submission(self):
        if not self.current_selection_coords:
            self._set_feedback_flash(self.COLOR_UI_FAILURE, 10)
            return -0.5 # Penalty for empty submission

        # Check if the sequence of coordinates matches any unfound word
        for word, data in self.word_data.items():
            if not data['found']:
                # Check both forward and reverse
                if self.current_selection_coords == data['coords'] or self.current_selection_coords == data['coords'][::-1]:
                    self.found_words.add(word)
                    data['found'] = True
                    self._set_feedback_flash(self.COLOR_UI_SUCCESS, 15)
                    # sound: word_found.wav
                    return 10.0 # Reward for finding a word
        
        # If no match was found, it's an incorrect submission
        self._set_feedback_flash(self.COLOR_UI_FAILURE, 10)
        # sound: incorrect.wav
        return -1.0 # Penalty for incorrect word

    def _set_feedback_flash(self, color, duration):
        self.feedback_flash_color = color
        self.feedback_flash_timer = duration

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # 1. Draw grid lines
        for i in range(self.grid_width + 1):
            x = self.grid_offset_x + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_pixel_height))
        for i in range(self.grid_height + 1):
            y = self.grid_offset_y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_pixel_width, y))

        # 2. Draw found word highlights
        for word, data in self.word_data.items():
            if data['found']:
                start_pos = (
                    self.grid_offset_x + data['coords'][0][0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.grid_offset_y + data['coords'][0][1] * self.CELL_SIZE + self.CELL_SIZE // 2
                )
                end_pos = (
                    self.grid_offset_x + data['coords'][-1][0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.grid_offset_y + data['coords'][-1][1] * self.CELL_SIZE + self.CELL_SIZE // 2
                )
                pygame.draw.line(self.screen, self.COLOR_UI_SUCCESS, start_pos, end_pos, 10)
        
        # 3. Draw letters
        found_coords = {coord for data in self.word_data.values() if data['found'] for coord in data['coords']}
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                letter = self.grid[y][x]
                color = self.COLOR_FOUND_LETTER if (x, y) in found_coords else self.COLOR_LETTER
                text_surf = self.font_letter.render(letter, True, color)
                text_rect = text_surf.get_rect(center=(
                    self.grid_offset_x + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.grid_offset_y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                ))
                self.screen.blit(text_surf, text_rect)

        # 4. Draw current selection path
        if len(self.current_selection_coords) > 1:
            points = []
            for x, y in self.current_selection_coords:
                points.append((
                    self.grid_offset_x + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.grid_offset_y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                ))
            pygame.draw.lines(self.screen, self.COLOR_SELECTION_PATH, False, points, 5)

        for x, y in self.current_selection_coords:
            center_x = self.grid_offset_x + x * self.CELL_SIZE + self.CELL_SIZE // 2
            center_y = self.grid_offset_y + y * self.CELL_SIZE + self.CELL_SIZE // 2
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, self.CELL_SIZE // 2 - 4, self.COLOR_SELECTION_PATH)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, self.CELL_SIZE // 2 - 4, self.COLOR_SELECTION_PATH)

        # 5. Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_offset_x + self.cursor_pos[0] * self.CELL_SIZE,
            self.grid_offset_y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
    
    def _render_ui(self):
        # 1. Draw Timer
        time_text = f"{max(0, int(self.time_left // 60)):02}:{max(0, int(self.time_left % 60)):02}"
        time_color = self.COLOR_UI_FAILURE if self.time_left < 10 and int(self.time_left * 2) % 2 == 0 else self.COLOR_UI_TEXT
        timer_surf = self.font_ui_large.render(time_text, True, time_color)
        self.screen.blit(timer_surf, (self.grid_offset_x, 5))
        
        # 2. Draw Words Found count
        found_text = f"FOUND: {len(self.found_words)}/{self.NUM_WORDS}"
        found_surf = self.font_ui_large.render(found_text, True, self.COLOR_UI_TEXT)
        found_rect = found_surf.get_rect(right=self.screen_width - 20, top=5)
        self.screen.blit(found_surf, found_rect)
        
        # 3. Draw Word List (to the right of the grid)
        list_x = self.grid_offset_x + self.grid_pixel_width + 20
        list_y = self.grid_offset_y
        for i, word in enumerate(self.words_to_find):
            if word in self.found_words:
                font = self.font_ui_strikethrough
                color = self.COLOR_UI_FOUND_WORD
            else:
                font = self.font_ui
                color = self.COLOR_UI_TEXT
            word_surf = font.render(word, True, color)
            self.screen.blit(word_surf, (list_x, list_y + i * 22))

        # 4. Draw current selection text
        current_word = "".join([self.grid[y][x] for x, y in self.current_selection_coords])
        if current_word:
            sel_surf = self.font_ui.render(f"Selected: {current_word}", True, self.COLOR_CURSOR)
            sel_rect = sel_surf.get_rect(centerx=self.screen_width / 2, bottom=self.screen_height - 10)
            self.screen.blit(sel_surf, sel_rect)
            
        # 5. Draw feedback flash
        if self.feedback_flash_timer > 0:
            flash_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            alpha = int(100 * (self.feedback_flash_timer / 15.0)) # Fade out
            flash_surface.fill((*self.feedback_flash_color, alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "found_words": len(self.found_words),
            "words_to_find": self.NUM_WORDS
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
        # Need to call reset first to initialize everything for rendering
        self.reset(seed=123)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset(seed=123)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")