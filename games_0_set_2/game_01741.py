
# Generated: 2025-08-28T02:35:08.461813
# Source Brief: brief_01741.md
# Brief Index: 1741

        
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
        "Controls: Use arrow keys to move the cursor. Press Space on a letter to start selecting a word, "
        "move to the end letter, and press Space again to submit. Press Shift to cancel your selection."
    )

    game_description = (
        "A classic word search puzzle. Find the 5 hidden words in the grid before the 90-second timer runs out. "
        "Correct words grant points, but incorrect submissions will penalize you."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 15
    CELL_SIZE = 24
    GRID_WIDTH = GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    MAX_TIME_SECONDS = 90
    FPS = 30

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_BG = (43, 48, 61)
    COLOR_GRID_LINES = (55, 62, 79)
    COLOR_LETTER = (210, 210, 210)
    COLOR_CURSOR = (76, 172, 255, 150)
    COLOR_SELECTION = (255, 229, 102)
    COLOR_SUCCESS = (118, 255, 122)
    COLOR_FAIL = (255, 107, 107)
    COLOR_FOUND_LINE = (118, 255, 122)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEADER = (150, 150, 150)
    COLOR_TIMER_DANGER = (255, 80, 80)

    WORD_LIST = [
        "PYTHON", "GYMNASIUM", "REWARD", "ACTION", "AGENT", "POLICY", "STATE",
        "PUZZLE", "SEARCH", "VISUAL", "PLAYER", "VECTOR", "MATRIX", "GRID",
        "LEARNING", "EPISODE", "TERMINAL", "STEP", "RESET", "RENDER", "FRAME",
        "ALGORITHM", "EXPLORE", "EXPLOIT", "NEURAL", "NETWORK", "TRAIN", "MODEL"
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
        
        self.font_letter = pygame.font.Font(None, 22)
        self.font_ui_header = pygame.font.Font(None, 24)
        self.font_ui_text = pygame.font.Font(None, 22)
        self.font_timer = pygame.font.Font(None, 48)
        self.font_strikethrough = pygame.font.Font(None, 22)
        self.font_strikethrough.set_strikethrough(True)

        self.grid_top_left = (
            (self.SCREEN_WIDTH - self.GRID_WIDTH) // 4,
            (self.SCREEN_HEIGHT - self.GRID_HEIGHT) // 2
        )

        self.np_random = None
        self._initialize_state()
        
        # This call validates the implementation against the spec
        self.validate_implementation()
    
    def _initialize_state(self):
        # Game state variables are initialized here and in reset()
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), "", dtype=str)
        self.solution = {}
        self.words_to_find = []
        self.found_words = set()
        self.found_word_paths = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.MAX_TIME_SECONDS * self.FPS
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selection_start = None
        self.prev_space_held = False
        self.prev_shift_held = False
        self.feedback_flash = {"color": None, "timer": 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()

        self._initialize_state()
        self._generate_grid()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Handle Input and State Update ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Update cursor position
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # Detect rising edge for space (press event)
        if space_held and not self.prev_space_held:
            # sound: "click"
            if self.selection_start is None:
                self.selection_start = tuple(self.cursor_pos)
                # Continuous reward for starting on a promising letter
                start_char = self.grid[self.selection_start[1], self.selection_start[0]]
                for word in self.words_to_find:
                    if word not in self.found_words and (word.startswith(start_char) or word.endswith(start_char)):
                        reward += 0.1
                        break
            else:
                submission_reward = self._submit_word()
                reward += submission_reward
        
        # Detect rising edge for shift (cancel event)
        if shift_held and not self.prev_shift_held:
            if self.selection_start is not None:
                # sound: "cancel"
                self.selection_start = None
                reward -= 0.5 # Small penalty for canceling

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # --- Update Game Logic ---
        self.steps += 1
        self.timer -= 1
        if self.feedback_flash["timer"] > 0:
            self.feedback_flash["timer"] -= 1

        # --- Check Termination ---
        all_words_found = len(self.found_words) == len(self.words_to_find)
        time_up = self.timer <= 0
        terminated = all_words_found or time_up

        if terminated and not self.game_over:
            self.game_over = True
            if all_words_found:
                # sound: "victory"
                reward += 50  # Goal-oriented reward

        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _submit_word(self):
        end_pos = tuple(self.cursor_pos)
        path = self._get_line_path(self.selection_start, end_pos)
        
        if not path:
            self.selection_start = None
            return -1

        word_fwd = "".join([self.grid[r, c] for c, r in path])
        word_bwd = word_fwd[::-1]
        
        found_match = False
        for potential_word in [word_fwd, word_bwd]:
            if potential_word in self.words_to_find and potential_word not in self.found_words:
                # sound: "success"
                self.found_words.add(potential_word)
                self.found_word_paths.append(path)
                self.feedback_flash = {"color": self.COLOR_SUCCESS, "timer": 15}
                self.selection_start = None
                return 10

        # sound: "fail"
        self.feedback_flash = {"color": self.COLOR_FAIL, "timer": 15}
        self.selection_start = None
        return -1

    def _get_line_path(self, start, end):
        # Bresenham's line algorithm for grid paths
        x0, y0 = start
        x1, y1 = end
        dx, dy = abs(x1 - x0), abs(y1 - y0)

        is_straight_line = (x0 == x1) or (y0 == y1) or (dx == dy)
        if not is_straight_line:
            return []

        path = []
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            path.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return path

    def _generate_grid(self):
        self.grid.fill(' ')
        self.solution = {}
        
        # Select 5 unique words that fit the grid
        valid_words = [w for w in self.WORD_LIST if len(w) <= self.GRID_SIZE]
        self.words_to_find = list(self.np_random.choice(valid_words, 5, replace=False))

        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]

        for word in self.words_to_find:
            placed = False
            for _ in range(100): # Max 100 attempts to place a word
                self.np_random.shuffle(directions)
                direction = directions[0]
                dr, dc = direction
                
                start_r = self.np_random.integers(0, self.GRID_SIZE)
                start_c = self.np_random.integers(0, self.GRID_SIZE)
                
                end_r = start_r + (len(word) - 1) * dr
                end_c = start_c + (len(word) - 1) * dc

                if not (0 <= end_r < self.GRID_SIZE and 0 <= end_c < self.GRID_SIZE):
                    continue

                can_place = True
                path = []
                for i in range(len(word)):
                    r, c = start_r + i * dr, start_c + i * dc
                    path.append((r, c))
                    if self.grid[r, c] != ' ' and self.grid[r, c] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i, (r, c) in enumerate(path):
                        self.grid[r, c] = word[i]
                    self.solution[word] = path
                    placed = True
                    break
            
            if not placed:
                # This is a fallback if placement fails, which is rare.
                # In a real product, you might regenerate the whole grid.
                # For this env, we just try again with a new set of words.
                return self._generate_grid()

        # Fill empty cells with random letters
        alphabet = string.ascii_uppercase
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == ' ':
                    self.grid[r, c] = self.np_random.choice(list(alphabet))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_ui()
        if self.feedback_flash["timer"] > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(150 * (self.feedback_flash["timer"] / 15))
            flash_surface.fill((*self.feedback_flash["color"], alpha))
            self.screen.blit(flash_surface, (0, 0))

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        gx, gy = self.grid_top_left
        grid_surface = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT))
        grid_surface.fill(self.COLOR_GRID_BG)

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(grid_surface, self.COLOR_GRID_LINES, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.GRID_HEIGHT))
            pygame.draw.line(grid_surface, self.COLOR_GRID_LINES, (0, i * self.CELL_SIZE), (self.GRID_WIDTH, i * self.CELL_SIZE))

        # Highlight current selection path
        if self.selection_start:
            path = self._get_line_path(self.selection_start, tuple(self.cursor_pos))
            for x, y in path:
                pygame.draw.rect(grid_surface, self.COLOR_SELECTION, (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Draw letters
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                letter_surf = self.font_letter.render(self.grid[r, c], True, self.COLOR_LETTER)
                letter_rect = letter_surf.get_rect(center=(c * self.CELL_SIZE + self.CELL_SIZE / 2, r * self.CELL_SIZE + self.CELL_SIZE / 2))
                grid_surface.blit(letter_surf, letter_rect)

        # Draw strikethroughs for found words
        for path in self.found_word_paths:
            start_pos = (path[0][0] * self.CELL_SIZE + self.CELL_SIZE // 2, path[0][1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            end_pos = (path[-1][0] * self.CELL_SIZE + self.CELL_SIZE // 2, path[-1][1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            pygame.draw.line(grid_surface, self.COLOR_FOUND_LINE, start_pos, end_pos, 4)

        # Draw cursor
        cursor_surf = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        cursor_surf.fill(self.COLOR_CURSOR)
        grid_surface.blit(cursor_surf, (self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE))

        self.screen.blit(grid_surface, self.grid_top_left)

    def _render_ui(self):
        ui_x = self.grid_top_left[0] + self.GRID_WIDTH + 30
        
        # --- Timer ---
        time_left_sec = math.ceil(self.timer / self.FPS)
        timer_color = self.COLOR_UI_TEXT if time_left_sec > 10 else self.COLOR_TIMER_DANGER
        timer_text = f"{time_left_sec}"
        timer_surf = self.font_timer.render(timer_text, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(timer_surf, timer_rect)

        # --- Score ---
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui_text.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 70))
        self.screen.blit(score_surf, score_rect)

        # --- Word List ---
        header_surf = self.font_ui_header.render("WORDS", True, self.COLOR_UI_HEADER)
        self.screen.blit(header_surf, (ui_x, 120))
        pygame.draw.line(self.screen, self.COLOR_UI_HEADER, (ui_x, 145), (ui_x + 100, 145), 1)

        y_offset = 160
        for word in self.words_to_find:
            font = self.font_strikethrough if word in self.found_words else self.font_ui_text
            color = self.COLOR_SUCCESS if word in self.found_words else self.COLOR_UI_TEXT
            word_surf = font.render(word, True, color)
            self.screen.blit(word_surf, (ui_x, y_offset))
            y_offset += 25

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_found": len(self.found_words),
            "words_remaining": len(self.words_to_find) - len(self.found_words)
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Word Search")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    # Store previous key states for edge detection
    key_states = {
        "space": False,
        "shift": False
    }

    while not terminated:
        # --- Human Controls ---
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        # --- Step the Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Score: {info['score']:.2f}, Reward: {reward:.2f}")

        # --- Render to Screen ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}")
    pygame.time.wait(2000)
    env.close()