
# Generated: 2025-08-28T03:35:26.894813
# Source Brief: brief_02069.md
# Brief Index: 2069

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select the first letter of a word. "
        "Move to the last letter and press Shift to submit the word."
    )

    game_description = (
        "A fast-paced word search puzzle. Find all 15 hidden words in the grid before the 3-minute timer runs out. "
        "Get points for finding words and a bonus for completing the puzzle."
    )

    auto_advance = True

    # --- Constants ---
    # Game parameters
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 20, 12
    NUM_WORDS_TO_FIND = 15
    TOTAL_TIME_SECONDS = 180
    FPS = 30
    MAX_STEPS = TOTAL_TIME_SECONDS * FPS

    # Colors
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_LINE = (40, 50, 60)
    COLOR_LETTER = (200, 210, 220)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION_START = (0, 150, 255, 100)
    COLOR_FOUND_WORD_LINE = (0, 255, 100)
    COLOR_FOUND_WORD_LETTER = (100, 120, 130)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_BG = (30, 40, 50)
    COLOR_FEEDBACK_CORRECT = (0, 255, 100, 50)
    COLOR_FEEDBACK_INCORRECT = (255, 50, 50, 50)

    # Word list for generation
    WORD_LIST = [
        "PYTHON", "AGENT", "REWARD", "ACTION", "POLICY", "STATE", "GRID", "SEARCH",
        "PUZZLE", "TIMER", "VISUAL", "LEARN", "DEEP", "NEURAL", "GAME", "VECTOR",
        "TENSOR", "FLUID", "MODEL", "TRAIN", "GYM", "SPACE", "RESET", "STEP", "CODE",
        "DEBUG", "KERNEL", "LAYER", "FRAME", "PIXEL", "MATRIX", "ALPHA", "BETA",
        "GAMMA", "DELTA", "EPOCH", "BATCH", "LOSS", "NOISE", "TARGET", "VALUE"
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

        # Fonts
        try:
            self.font_letter = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_ui = pygame.font.SysFont("Arial", 22, bold=True)
        except pygame.error:
            self.font_letter = pygame.font.Font(None, 24)
            self.font_ui = pygame.font.Font(None, 26)

        # Dynamic layout calculations
        self.grid_area_height = self.SCREEN_HEIGHT - 50
        self.cell_width = self.SCREEN_WIDTH // self.GRID_COLS
        self.cell_height = self.grid_area_height // self.GRID_ROWS
        self.grid_offset_x = (self.SCREEN_WIDTH - self.GRID_COLS * self.cell_width) // 2
        self.grid_offset_y = (self.grid_area_height - self.GRID_ROWS * self.cell_height) // 2

        # State variables (initialized in reset)
        self.grid = None
        self.word_data = None
        self.word_lookup = None
        self.is_part_of_word_grid = None
        self.cursor_pos = None
        self.selection_start = None
        self.found_words_info = None
        self.time_remaining = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.previous_action = None
        self.feedback_effect = None

        self.reset()
        self.validate_implementation()

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.word_data = []
        words_to_place = self.np_random.choice(self.WORD_LIST, self.NUM_WORDS_TO_FIND, replace=False).tolist()
        
        directions = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]

        for word in words_to_place:
            placed = False
            for _ in range(100): # 100 placement attempts
                word_to_try = word if self.np_random.random() < 0.5 else word[::-1]
                l = len(word_to_try)
                d = self.np_random.choice(len(directions))
                dx, dy = directions[d]

                start_x = self.np_random.integers(0, self.GRID_COLS)
                start_y = self.np_random.integers(0, self.GRID_ROWS)
                
                end_x, end_y = start_x + (l - 1) * dx, start_y + (l - 1) * dy

                if not (0 <= end_x < self.GRID_COLS and 0 <= end_y < self.GRID_ROWS):
                    continue

                can_place = True
                for i in range(l):
                    x, y = start_x + i * dx, start_y + i * dy
                    if self.grid[y][x] != '' and self.grid[y][x] != word_to_try[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i in range(l):
                        x, y = start_x + i * dx, start_y + i * dy
                        self.grid[y][x] = word_to_try[i]
                    
                    self.word_data.append({
                        "word": word,
                        "start_pos": (start_x, start_y),
                        "end_pos": (end_x, end_y)
                    })
                    placed = True
                    break
            if not placed:
                # This should be rare with a reasonable grid size and word list
                # For robustness, we could try again with a different word
                pass

        # Fill empty cells with random letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == '':
                    self.grid[r][c] = self.np_random.choice(list(string.ascii_uppercase))

        # Create lookup tables for performance
        self.word_lookup = {}
        self.is_part_of_word_grid = [[False for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        for data in self.word_data:
            start, end = data["start_pos"], data["end_pos"]
            self.word_lookup[(start, end)] = data["word"]
            self.word_lookup[(end, start)] = data["word"]
            
            dx = np.sign(end[0] - start[0])
            dy = np.sign(end[1] - start[1])
            for i in range(len(data["word"])):
                x, y = start[0] + i * dx, start[1] + i * dy
                self.is_part_of_word_grid[y][x] = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.TOTAL_TIME_SECONDS
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selection_start = None
        self.found_words_info = []
        self.previous_action = [0, 0, 0]
        self.feedback_effect = None

        self._generate_grid()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        self.time_remaining -= 1.0 / self.FPS
        
        # --- Handle Actions ---
        movement, space_val, shift_val = action[0], action[1], action[2]
        space_pressed = space_val == 1 and self.previous_action[1] == 0
        shift_pressed = shift_val == 1 and self.previous_action[2] == 0
        self.previous_action = action

        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

        # Spacebar: Select start letter
        if space_pressed:
            if self.selection_start is None:
                self.selection_start = tuple(self.cursor_pos)
            else: # Pressing space again cancels selection
                self.selection_start = None

        # Shift: Select end letter and submit
        if shift_pressed and self.selection_start is not None:
            end_pos = tuple(self.cursor_pos)
            submission_result = self._check_word_submission(self.selection_start, end_pos)
            reward += submission_result
            self.selection_start = None

        # Continuous reward for cursor position
        cx, cy = self.cursor_pos
        is_on_word = self.is_part_of_word_grid[cy][cx]
        is_found = any(self._is_coord_in_found_word(cx, cy, info) for info in self.found_words_info)
        if is_on_word and not is_found:
            reward += 0.01 # Small reward for exploring correctly
        else:
            reward -= 0.01 # Small penalty for being on empty/found letters

        self.score += reward
        self.steps += 1
        
        # --- Termination ---
        terminated = (self.time_remaining <= 0) or (len(self.found_words_info) >= self.NUM_WORDS_TO_FIND) or (self.steps >= self.MAX_STEPS)
        if terminated and len(self.found_words_info) >= self.NUM_WORDS_TO_FIND:
            win_bonus = 50.0
            reward += win_bonus
            self.score += win_bonus

        # Update feedback effect timer
        if self.feedback_effect:
            effect_type, duration = self.feedback_effect
            if duration > 0:
                self.feedback_effect = (effect_type, duration - 1)
            else:
                self.feedback_effect = None
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _is_coord_in_found_word(self, x, y, word_info):
        start, end = word_info["start_pos"], word_info["end_pos"]
        l = len(word_info["word"])
        dx = np.sign(end[0] - start[0])
        dy = np.sign(end[1] - start[1])
        for i in range(l):
            if (start[0] + i * dx, start[1] + i * dy) == (x, y):
                return True
        return False

    def _check_word_submission(self, start_pos, end_pos):
        submitted_tuple = (start_pos, end_pos)
        if submitted_tuple in self.word_lookup:
            word_str = self.word_lookup[submitted_tuple]
            # Check if already found
            if any(info["word"] == word_str for info in self.found_words_info):
                self.feedback_effect = ('incorrect', self.FPS // 2) # 0.5s flash
                return 0

            # Find original data to store
            for data in self.word_data:
                if data["word"] == word_str:
                    self.found_words_info.append(data)
                    break
            
            self.feedback_effect = ('correct', self.FPS // 2)
            return 10.0 # Reward for finding a word
        else:
            self.feedback_effect = ('incorrect', self.FPS // 2)
            return 0

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                letter_surf = self.font_letter.render(self.grid[r][c], True, self.COLOR_LETTER)
                letter_rect = letter_surf.get_rect(center=(
                    self.grid_offset_x + c * self.cell_width + self.cell_width // 2,
                    self.grid_offset_y + r * self.cell_height + self.cell_height // 2
                ))
                self.screen.blit(letter_surf, letter_rect)

        # Draw highlights for found words
        for info in self.found_words_info:
            start_pos, end_pos = info["start_pos"], info["end_pos"]
            start_center = (
                self.grid_offset_x + start_pos[0] * self.cell_width + self.cell_width // 2,
                self.grid_offset_y + start_pos[1] * self.cell_height + self.cell_height // 2
            )
            end_center = (
                self.grid_offset_x + end_pos[0] * self.cell_width + self.cell_width // 2,
                self.grid_offset_y + end_pos[1] * self.cell_height + self.cell_height // 2
            )
            pygame.draw.line(self.screen, self.COLOR_FOUND_WORD_LINE, start_center, end_center, 4)
            # Dim the letters of found words
            dx = np.sign(end_pos[0] - start_pos[0])
            dy = np.sign(end_pos[1] - start_pos[1])
            for i in range(len(info["word"])):
                x, y = start_pos[0] + i * dx, start_pos[1] + i * dy
                letter_surf = self.font_letter.render(self.grid[y][x], True, self.COLOR_FOUND_WORD_LETTER)
                letter_rect = letter_surf.get_rect(center=(
                    self.grid_offset_x + x * self.cell_width + self.cell_width // 2,
                    self.grid_offset_y + y * self.cell_height + self.cell_height // 2
                ))
                self.screen.blit(letter_surf, letter_rect)

        # Draw selection start highlight
        if self.selection_start is not None:
            c, r = self.selection_start
            highlight_rect = pygame.Rect(
                self.grid_offset_x + c * self.cell_width,
                self.grid_offset_y + r * self.cell_height,
                self.cell_width, self.cell_height
            )
            s = pygame.Surface((self.cell_width, self.cell_height), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECTION_START)
            self.screen.blit(s, highlight_rect.topleft)

        # Draw cursor
        c, r = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.grid_offset_x + c * self.cell_width,
            self.grid_offset_y + r * self.cell_height,
            self.cell_width, self.cell_height
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Draw feedback effect
        if self.feedback_effect:
            effect_type, _ = self.feedback_effect
            color = self.COLOR_FEEDBACK_CORRECT if effect_type == 'correct' else self.COLOR_FEEDBACK_INCORRECT
            s = pygame.Surface((self.SCREEN_WIDTH, self.grid_area_height), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, (0, 0))

    def _render_ui(self):
        ui_bar = pygame.Rect(0, self.SCREEN_HEIGHT - 50, self.SCREEN_WIDTH, 50)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_bar)
        pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (0, self.SCREEN_HEIGHT - 50), (self.SCREEN_WIDTH, self.SCREEN_HEIGHT - 50), 2)

        # Timer
        mins, secs = divmod(max(0, int(self.time_remaining)), 60)
        timer_text = f"TIME: {mins:02d}:{secs:02d}"
        timer_surf = self.font_ui.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, (20, self.SCREEN_HEIGHT - 35))

        # Word count
        word_count_text = f"FOUND: {len(self.found_words_info)} / {self.NUM_WORDS_TO_FIND}"
        word_count_surf = self.font_ui.render(word_count_text, True, self.COLOR_UI_TEXT)
        word_count_rect = word_count_surf.get_rect(centerx=self.SCREEN_WIDTH // 2, y=self.SCREEN_HEIGHT - 35)
        self.screen.blit(word_count_surf, word_count_rect)

        # Score
        score_text = f"SCORE: {int(self.score)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(right=self.SCREEN_WIDTH - 20, y=self.SCREEN_HEIGHT - 35)
        self.screen.blit(score_surf, score_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "words_found": len(self.found_words_info),
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Search Puzzle")
    clock = pygame.time.Clock()
    
    print(env.game_description)
    print(env.user_guide)

    running = True
    while running:
        # --- Action mapping from keyboard to MultiDiscrete action ---
        movement = 0 # No-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            # print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}")
            pass

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Words Found: {info['words_found']}")
            obs, info = env.reset()

        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()