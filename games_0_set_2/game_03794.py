
# Generated: 2025-08-28T00:27:14.738972
# Source Brief: brief_03794.md
# Brief Index: 3794

        
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
    """
    A Gymnasium environment for a word search game.
    The agent controls a cursor on a grid of letters and must find hidden words
    by selecting letters in sequence before a timer runs out.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Hold Space to select letters. Press Shift to submit your word."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find hidden words in a grid of letters before time runs out. Select letters by moving the cursor and holding space, then press shift to submit."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # Word list for the game
    WORD_BANK = [
        "PYTHON", "GYM", "AGENT", "REWARD", "ACTION", "STATE", "POLICY", "LEARN",
        "GRID", "SEARCH", "PUZZLE", "SOLVE", "CODE", "GAME", "VECTOR", "TENSOR",
        "MODEL", "NEURAL", "DEEP", "PIXEL", "VISUAL", "PLAY", "STEP", "RESET",
        "SPACE", "BOX", "DISCRETE", "ENV", "NUMPY", "PYGAME"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 12
        self.CELL_SIZE = 30
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_OFFSET_X = 20
        self.GRID_OFFSET_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2
        self.UI_X_START = self.GRID_OFFSET_X + self.GRID_WIDTH + 20
        self.MAX_STEPS = 3000
        self.WORDS_TO_FIND = 10

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID_LINES = (40, 50, 60)
        self.COLOR_LETTER = (200, 210, 220)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_SELECTION = (255, 200, 0, 100)
        self.COLOR_FOUND_WORD_GRID = (0, 255, 100, 80)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_TITLE = (255, 200, 0)
        self.COLOR_UI_FOUND = (100, 120, 140)
        self.COLOR_FEEDBACK_CORRECT = (0, 255, 100, 150)
        self.COLOR_FEEDBACK_INCORRECT = (255, 50, 50, 150)

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
        self.font_letter = pygame.font.Font(None, 28)
        self.font_ui_text = pygame.font.Font(None, 20)
        self.font_ui_title = pygame.font.Font(None, 24)
        self.font_ui_score = pygame.font.Font(None, 32)
        
        self.np_random = None

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.grid = []
        self.word_list = []
        self.word_metadata = {}
        self.found_words = set()
        self.cursor_pos = [0, 0]
        self.selection_active = False
        self.selected_path = []
        self.feedback_flash = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selection_active = False
        self.selected_path = []
        self.found_words = set()
        self.feedback_flash = None

        # Procedurally generate the puzzle
        puzzle_generated = False
        while not puzzle_generated:
            puzzle_generated, self.grid, self.word_list, self.word_metadata = self._generate_puzzle()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.01  # Small penalty for each step to encourage speed

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 2:  # Down
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_SIZE
        elif movement == 3:  # Left
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_SIZE) % self.GRID_SIZE
        elif movement == 4:  # Right
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_SIZE
        
        # 2. Handle word submission (Shift takes precedence)
        if shift_held:
            if self.selection_active:
                # sound: submit_word.wav
                word = "".join([self.grid[r][c] for r, c in self.selected_path])
                
                is_correct = word in self.word_list and word not in self.found_words
                is_correct_reversed = word[::-1] in self.word_list and word[::-1] not in self.found_words
                
                if is_correct or is_correct_reversed:
                    # sound: correct_word.wav
                    actual_word = word if is_correct else word[::-1]
                    self.found_words.add(actual_word)
                    reward += 10
                    self.score += 100 * len(actual_word)
                    self.feedback_flash = {"color": self.COLOR_FEEDBACK_CORRECT, "timer": 15}
                else:
                    # sound: incorrect_word.wav
                    reward -= 2
                    self.score -= 50
                    self.feedback_flash = {"color": self.COLOR_FEEDBACK_INCORRECT, "timer": 15}
                
                # Reset selection after submission
                self.selection_active = False
                self.selected_path = []
        
        # 3. Handle letter selection
        elif space_held:
            r, c = self.cursor_pos
            pos = (r, c)
            if not self.selection_active:
                # sound: select_start.wav
                self.selection_active = True
                self.selected_path = [pos]
            elif pos != self.selected_path[-1]:
                # sound: select_extend.wav
                last_r, last_c = self.selected_path[-1]
                # Allow adding if adjacent (incl. diagonals)
                if abs(r - last_r) <= 1 and abs(c - last_c) <= 1:
                    if pos not in self.selected_path:
                        self.selected_path.append(pos)
                else: # Invalid (non-contiguous) selection, reset path
                    self.selection_active = False
                    self.selected_path = []


        # 4. Check for termination
        terminated = False
        if len(self.found_words) == len(self.word_list):
            # sound: victory.wav
            reward += 50
            self.score += 1000
            terminated = True
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            # sound: timeout.wav
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _generate_puzzle(self):
        grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        # Select words
        words_to_place = self.np_random.choice(self.WORD_BANK, size=self.WORDS_TO_FIND, replace=False).tolist()
        words_to_place.sort(key=len, reverse=True) # Place longer words first
        
        word_metadata = {}
        
        directions = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]
        
        for word in words_to_place:
            placed = False
            for _ in range(100): # 100 attempts to place a word
                shuffled_directions = list(directions)
                self.np_random.shuffle(shuffled_directions)
                
                direction = shuffled_directions[0]
                dr, dc = direction[0], direction[1]
                
                # Choose random starting position
                start_r = self.np_random.integers(0, self.GRID_SIZE)
                start_c = self.np_random.integers(0, self.GRID_SIZE)
                
                end_r = start_r + (len(word) - 1) * dr
                end_c = start_c + (len(word) - 1) * dc
                
                # Check if word fits in grid
                if not (0 <= end_r < self.GRID_SIZE and 0 <= end_c < self.GRID_SIZE):
                    continue
                
                # Check for conflicts
                can_place = True
                temp_path = []
                for i in range(len(word)):
                    r, c = start_r + i * dr, start_c + i * dc
                    temp_path.append((r,c))
                    if grid[r][c] != '' and grid[r][c] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i in range(len(word)):
                        r, c = start_r + i * dr, start_c + i * dc
                        grid[r][c] = word[i]
                    word_metadata[word] = temp_path
                    placed = True
                    break
            
            if not placed:
                return False, None, None, None # Failed to generate, try again

        # Fill remaining grid with random letters
        alphabet = string.ascii_uppercase
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if grid[r][c] == '':
                    grid[r][c] = self.np_random.choice(list(alphabet))
                    
        return True, grid, words_to_place, word_metadata

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid_and_words()
        self._render_selection_and_cursor()
        self._render_ui()
        
        # Handle feedback flash
        if self.feedback_flash:
            flash_surface = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.feedback_flash["color"])
            self.screen.blit(flash_surface, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y))
            self.feedback_flash["timer"] -= 1
            if self.feedback_flash["timer"] <= 0:
                self.feedback_flash = None

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid_and_words(self):
        # Draw found words background
        for word in self.found_words:
            if word in self.word_metadata:
                path = self.word_metadata[word]
                for r, c in path:
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + c * self.CELL_SIZE,
                        self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    pygame.gfxdraw.box(self.screen, rect, self.COLOR_FOUND_WORD_GRID)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_WIDTH, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, start_pos, end_pos)

        # Draw letters
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                letter = self.grid[r][c]
                text_surf = self.font_letter.render(letter, True, self.COLOR_LETTER)
                text_rect = text_surf.get_rect(center=(
                    self.GRID_OFFSET_X + c * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.GRID_OFFSET_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                ))
                self.screen.blit(text_surf, text_rect)

    def _render_selection_and_cursor(self):
        # Draw current selection
        if self.selection_active and self.selected_path:
            selection_surface = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            for r, c in self.selected_path:
                rect = pygame.Rect(
                    self.GRID_OFFSET_X + c * self.CELL_SIZE,
                    self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.gfxdraw.box(selection_surface, rect, self.COLOR_SELECTION)
            self.screen.blit(selection_surface, (0,0))

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_OFFSET_X + cursor_c * self.CELL_SIZE,
            self.GRID_OFFSET_Y + cursor_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)


    def _render_ui(self):
        # --- Word List ---
        title_surf = self.font_ui_title.render("WORDS TO FIND", True, self.COLOR_UI_TITLE)
        self.screen.blit(title_surf, (self.UI_X_START, 20))
        
        y_offset = 50
        for word in self.word_list:
            color = self.COLOR_UI_FOUND if word in self.found_words else self.COLOR_UI_TEXT
            word_surf = self.font_ui_text.render(word, True, color)
            self.screen.blit(word_surf, (self.UI_X_START + 10, y_offset))
            if word in self.found_words:
                # Strikethrough
                line_y = y_offset + word_surf.get_height() // 2
                pygame.draw.line(self.screen, self.COLOR_UI_FOUND, 
                                 (self.UI_X_START + 8, line_y), 
                                 (self.UI_X_START + 12 + word_surf.get_width(), line_y), 2)
            y_offset += 20

        # --- Score and Time ---
        score_y = self.HEIGHT - 80
        time_y = self.HEIGHT - 40

        # Score
        score_title_surf = self.font_ui_text.render("SCORE", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_title_surf, (self.UI_X_START, score_y - 20))
        score_val_surf = self.font_ui_score.render(f"{self.score}", True, self.COLOR_UI_TITLE)
        self.screen.blit(score_val_surf, (self.UI_X_START, score_y))

        # Time
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_title_surf = self.font_ui_text.render("TIME LEFT", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_title_surf, (self.UI_X_START, time_y - 20))
        time_val_surf = self.font_ui_score.render(f"{time_left}", True, self.COLOR_UI_TITLE)
        self.screen.blit(time_val_surf, (self.UI_X_START, time_y))


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_found": len(self.found_words),
            "words_total": len(self.word_list),
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

if __name__ == "__main__":
    # To run and play the game
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Word Search Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    game_paused = False

    while running:
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    print("--- Resetting Environment ---")
                    obs, info = env.reset()
                    total_reward = 0
                    game_paused = False
                if env.game_over:
                    game_paused = True

        if game_paused:
            continue

        # --- Continuous key presses to form an action ---
        action = np.array([0, 0, 0]) # Default no-op
        keys = pygame.key.get_pressed()
        
        # Movement (only one direction at a time)
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        # Buttons
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # --- Render the observation to the screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated and not game_paused:
            print(f"--- Episode Finished ---")
            print(f"Final Score: {info['score']}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Steps: {info['steps']}")
            print("Press 'R' to play again or close the window.")
            game_paused = True
            
        clock.tick(30) # Limit human play speed

    env.close()