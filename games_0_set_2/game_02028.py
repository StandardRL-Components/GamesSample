
# Generated: 2025-08-27T19:01:39.635651
# Source Brief: brief_02028.md
# Brief Index: 2028

        
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
        "Controls: ↑↓←→ to move the cursor. Press space to select/deselect a letter. Hold shift to submit the current selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find hidden words in a grid of letters before the timer runs out. Select letters one by one and submit your guess."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 16, 11
    GRID_AREA_WIDTH, GRID_AREA_HEIGHT = 400, 330
    CELL_SIZE = 30
    GRID_TOP_LEFT = (
        (GRID_AREA_WIDTH - GRID_COLS * CELL_SIZE) // 2 + 20,
        (SCREEN_HEIGHT - GRID_ROWS * CELL_SIZE) // 2,
    )

    MAX_TIME = 60.0  # seconds
    MAX_STEPS = 6000 # 100 FPS for 60 seconds

    WORD_COUNT = 10
    MASTER_WORD_LIST = [
        "PYTHON", "AGENT", "REWARD", "STATE", "ACTION", "POLICY", "LEARN",
        "GRID", "WORLD", "SPACE", "VECTOR", "TENSOR", "MODEL", "SOLVE",
        "PUZZLE", "SEARCH", "FIND", "CODE", "GAME", "FRAME", "PIXEL", "GYM",
        "RENDER", "RESET", "STEP", "VISUAL", "PLAY", "TIMER", "SCORE", "WIN"
    ]

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 45, 60)
    COLOR_GRID_LINE = (40, 60, 80)
    COLOR_LETTER = (200, 210, 220)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTED_BG = (0, 100, 200, 180)
    COLOR_FOUND_BG = (0, 150, 50, 180)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEADER = (255, 200, 0)
    COLOR_WORD_FOUND = (100, 120, 140)
    COLOR_SUCCESS = (0, 255, 0)
    COLOR_FAIL = (255, 0, 0)
    COLOR_TIMER_BAR = (0, 150, 255)
    COLOR_TIMER_BAR_WARN = (255, 80, 0)


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
        try:
            self.font_letter = pygame.font.SysFont("dejavusansmono", 18, bold=True)
            self.font_ui_text = pygame.font.SysFont("dejavusans", 16)
            self.font_ui_header = pygame.font.SysFont("dejavusans", 18, bold=True)
        except pygame.error:
            self.font_letter = pygame.font.SysFont("monospace", 18, bold=True)
            self.font_ui_text = pygame.font.SysFont("sans", 16)
            self.font_ui_header = pygame.font.SysFont("sans", 18, bold=True)

        # Initialize state variables
        self.grid = []
        self.words_to_find = {}
        self.cursor_pos = [0, 0]
        self.selected_path = []
        self.correct_letter_coords = set()
        self.timer = 0.0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.prev_shift_held = False
        self.feedback_flash_timer = 0
        self.feedback_flash_color = (0,0,0)

        self.reset()
        self.validate_implementation()

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        words = self.np_random.choice(self.MASTER_WORD_LIST, self.WORD_COUNT, replace=False).tolist()
        self.words_to_find = {word: {"found": False, "coords": []} for word in words}

        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]

        for word in self.words_to_find.keys():
            placed = False
            for _ in range(100): # Max placement attempts
                direction = self.np_random.choice(list(range(len(directions))))
                dr, dc = directions[direction]
                
                start_r = self.np_random.integers(0, self.GRID_ROWS)
                start_c = self.np_random.integers(0, self.GRID_COLS)

                end_r = start_r + (len(word) - 1) * dr
                end_c = start_c + (len(word) - 1) * dc

                if not (0 <= end_r < self.GRID_ROWS and 0 <= end_c < self.GRID_COLS):
                    continue

                can_place = True
                path = []
                for i in range(len(word)):
                    r, c = start_r + i * dr, start_c + i * dc
                    path.append((r, c))
                    if self.grid[r][c] != '' and self.grid[r][c] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i in range(len(word)):
                        r, c = path[i]
                        self.grid[r][c] = word[i]
                    self.words_to_find[word]["coords"] = path
                    placed = True
                    break
            if not placed:
                # If a word can't be placed, restart the whole generation.
                return self._generate_grid()

        # Fill empty cells with random letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == '':
                    self.grid[r][c] = self.np_random.choice(list(alphabet))
        
        # Pre-calculate all correct letter coordinates for reward shaping
        self.correct_letter_coords = set()
        for data in self.words_to_find.values():
            self.correct_letter_coords.update(data["coords"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME
        
        self._generate_grid()

        self.cursor_pos = [self.GRID_ROWS // 2, self.GRID_COLS // 2]
        self.selected_path = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.feedback_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle Actions ---
        # Movement
        if movement == 1: self.cursor_pos[0] -= 1  # Up
        elif movement == 2: self.cursor_pos[0] += 1  # Down
        elif movement == 3: self.cursor_pos[1] -= 1  # Left
        elif movement == 4: self.cursor_pos[1] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_ROWS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_COLS - 1)

        # Selection (Space press)
        if space_held and not self.prev_space_held:
            # Sfx: select_click.wav
            pos = tuple(self.cursor_pos)
            if pos in self.selected_path:
                # If re-clicking a letter, deselect from that point onwards
                idx = self.selected_path.index(pos)
                self.selected_path = self.selected_path[:idx]
            else:
                self.selected_path.append(pos)
                # Continuous reward for selecting a potentially correct letter
                if pos in self.correct_letter_coords:
                    reward += 0.1
                else:
                    reward -= 0.1

        # Submission (Shift press)
        if shift_held and not self.prev_shift_held and self.selected_path:
            selected_word = "".join([self.grid[r][c] for r, c in self.selected_path])
            found_match = False
            
            for word, data in self.words_to_find.items():
                if not data["found"] and (word == selected_word or word == selected_word[::-1]):
                    # Sfx: success.wav
                    data["found"] = True
                    self.score += 10
                    reward += 10
                    found_match = True
                    self.feedback_flash_timer = 15 # frames
                    self.feedback_flash_color = self.COLOR_SUCCESS
                    break
            
            if not found_match:
                # Sfx: fail.wav
                self.feedback_flash_timer = 15
                self.feedback_flash_color = self.COLOR_FAIL

            self.selected_path = []


        # --- Update State ---
        self.steps += 1
        self.timer = max(0, self.MAX_TIME - (self.steps / self.MAX_STEPS) * self.MAX_TIME)
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if self.feedback_flash_timer > 0:
            self.feedback_flash_timer -= 1

        # --- Check Termination ---
        words_found_count = sum(1 for data in self.words_to_find.values() if data["found"])
        all_words_found = words_found_count == self.WORD_COUNT
        
        if all_words_found:
            reward += 50 # Bonus for completing the game
            self.game_over = True
            # Sfx: victory_fanfare.wav
        
        if self.timer <= 0:
            self.game_over = True

        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(
            self.GRID_TOP_LEFT[0], self.GRID_TOP_LEFT[1],
            self.GRID_COLS * self.CELL_SIZE, self.GRID_ROWS * self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)
        
        # Pre-calculate highlight surfaces
        found_coords = set()
        for data in self.words_to_find.values():
            if data["found"]:
                found_coords.update(data["coords"])

        # Draw highlights and letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                cell_x = self.GRID_TOP_LEFT[0] + c * self.CELL_SIZE
                cell_y = self.GRID_TOP_LEFT[1] + r * self.CELL_SIZE
                
                # Draw found word background
                if (r, c) in found_coords:
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    s.fill(self.COLOR_FOUND_BG)
                    self.screen.blit(s, (cell_x, cell_y))
                
                # Draw current selection background
                if (r, c) in self.selected_path:
                    s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                    s.fill(self.COLOR_SELECTED_BG)
                    self.screen.blit(s, (cell_x, cell_y))
                
                # Draw letter
                letter_surf = self.font_letter.render(self.grid[r][c], True, self.COLOR_LETTER)
                letter_rect = letter_surf.get_rect(center=(cell_x + self.CELL_SIZE // 2, cell_y + self.CELL_SIZE // 2))
                self.screen.blit(letter_surf, letter_rect)

        # Draw grid lines
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_TOP_LEFT[1] + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_TOP_LEFT[0], y), (grid_rect.right, y))
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_TOP_LEFT[0] + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.GRID_TOP_LEFT[1]), (x, grid_rect.bottom))

        # Draw cursor
        cursor_x = self.GRID_TOP_LEFT[0] + self.cursor_pos[1] * self.CELL_SIZE
        cursor_y = self.GRID_TOP_LEFT[1] + self.cursor_pos[0] * self.CELL_SIZE
        cursor_rect = pygame.Rect(cursor_x, cursor_y, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Draw feedback flash
        if self.feedback_flash_timer > 0:
            flash_alpha = int(128 * (self.feedback_flash_timer / 15))
            flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            flash_surface.fill((*self.feedback_flash_color, flash_alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # --- Timer Bar ---
        timer_percent = self.timer / self.MAX_TIME
        bar_color = self.COLOR_TIMER_BAR if timer_percent > 0.25 else self.COLOR_TIMER_BAR_WARN
        bar_width = int(self.SCREEN_WIDTH * timer_percent)
        pygame.draw.rect(self.screen, bar_color, (0, 0, bar_width, 10))

        # --- Right Panel (Words to Find & Score) ---
        panel_x = self.GRID_TOP_LEFT[0] + self.GRID_COLS * self.CELL_SIZE + 20
        
        # Score
        score_header = self.font_ui_header.render("SCORE", True, self.COLOR_UI_HEADER)
        self.screen.blit(score_header, (panel_x, 30))
        score_text = self.font_ui_header.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (panel_x, 50))
        
        # Words List
        words_header = self.font_ui_header.render("WORDS", True, self.COLOR_UI_HEADER)
        self.screen.blit(words_header, (panel_x, 100))
        
        y_offset = 125
        sorted_words = sorted(self.words_to_find.keys())
        for word in sorted_words:
            data = self.words_to_find[word]
            color = self.COLOR_WORD_FOUND if data["found"] else self.COLOR_UI_TEXT
            word_surf = self.font_ui_text.render(word, True, color)
            self.screen.blit(word_surf, (panel_x, y_offset))
            if data["found"]:
                # Draw strikethrough
                line_y = y_offset + word_surf.get_height() // 2
                pygame.draw.line(self.screen, self.COLOR_WORD_FOUND, (panel_x, line_y), (panel_x + word_surf.get_width(), line_y), 2)
            y_offset += 20
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "words_found": sum(1 for d in self.words_to_find.values() if d["found"])
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
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for manual play ---
    pygame.display.set_caption("Word Search")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    
    while running:
        # --- Action mapping for human player ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESET ---")
                obs, info = env.reset()
                total_reward = 0

        # --- Step the environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Optional: auto-reset after a delay
            pygame.time.wait(2000)
            print("--- RESET ---")
            obs, info = env.reset()
            total_reward = 0

        # --- Render the observation to the display window ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        clock.tick(100) # Match the step rate
        
    env.close()