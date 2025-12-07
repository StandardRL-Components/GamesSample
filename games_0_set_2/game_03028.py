
# Generated: 2025-08-27T22:08:48.647514
# Source Brief: brief_03028.md
# Brief Index: 3028

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Use arrow keys to move the cursor. Press Space to start a selection, "
        "move the cursor to the end of the word, and press Space again to submit. "
        "Press Shift to cancel a selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A timed word search puzzle. Find all the hidden words in the grid "
        "by selecting them before the 60-second timer runs out."
    )

    # Frames advance on action, not automatically.
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    CELL_SIZE = 20
    GRID_WIDTH = GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    UI_WIDTH = SCREEN_WIDTH - GRID_WIDTH
    MAX_TIME_STEPS = 1800  # 60 seconds at 30fps if auto-advancing, now just a step limit
    WORDS_TO_FIND = 15

    # --- Colors ---
    COLOR_BG = (240, 248, 255)  # AliceBlue
    COLOR_GRID_LINES = (211, 211, 211)  # LightGray
    COLOR_LETTER = (25, 25, 112)  # MidnightBlue
    COLOR_CURSOR = (255, 165, 0, 150)  # Orange, semi-transparent
    COLOR_SELECTION = (60, 179, 113, 100)  # MediumSeaGreen, semi-transparent
    COLOR_TEXT = (47, 79, 79)  # DarkSlateGray
    COLOR_SUCCESS_FLASH = (0, 255, 0)
    COLOR_FAIL_FLASH = (255, 0, 0)
    COLOR_WORD_FOUND = (169, 169, 169) # DarkGray

    # --- Word Bank ---
    WORD_BANK = [
        "PYTHON", "GYMNASIUM", "REWARD", "ACTION", "AGENT", "POLICY", "STATE",
        "EPISODE", "LEARNING", "NEURAL", "NETWORK", "VECTOR", "TENSOR", "FRAME",
        "RENDER", "SEARCH", "GRID", "PUZZLE", "SOLVE", "TIMER", "RESET", "STEP",
        "VISUAL", "EXPERT", "PLAYER", "GAME", "CODE", "DEBUG", "SPACE", "ARRAY"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_letter = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_ui_title = pygame.font.SysFont("Verdana", 16, bold=True)
        self.font_ui_word = pygame.font.SysFont("Verdana", 14)
        self.font_ui_strikethrough = pygame.font.SysFont("Verdana", 14)
        self.font_ui_strikethrough.set_strikethrough(True)

        # Game state variables (initialized in reset)
        self.grid = None
        self.word_list = None
        self.word_locations = None
        self.found_words = None
        self.cursor_pos = None
        self.selection_start = None
        self.previous_action = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.visual_effects = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selection_start = None
        self.previous_action = self.action_space.sample() * 0 # All zeros
        self.visual_effects = [] # List of (type, pos, timer)

        self._generate_new_puzzle()

        return self._get_observation(), self._get_info()

    def _generate_new_puzzle(self):
        # Select words
        self.word_list = self.np_random.choice(self.WORD_BANK, self.WORDS_TO_FIND, replace=False).tolist()
        self.word_list.sort(key=len, reverse=True) # Place longer words first
        self.found_words = set()
        self.word_locations = {}

        # Initialize grid
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]

        # Place words
        for word in self.word_list:
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                attempts += 1
                direction = self.np_random.integers(0, 8) # 8 directions
                dx = [0, 1, 1, 1, 0, -1, -1, -1][direction]
                dy = [-1, -1, 0, 1, 1, 1, 0, -1][direction]

                x_start = self.np_random.integers(0, self.GRID_SIZE)
                y_start = self.np_random.integers(0, self.GRID_SIZE)

                x_end = x_start + (len(word) - 1) * dx
                y_end = y_start + (len(word) - 1) * dy

                if 0 <= x_end < self.GRID_SIZE and 0 <= y_end < self.GRID_SIZE:
                    can_place = True
                    for i in range(len(word)):
                        x, y = x_start + i * dx, y_start + i * dy
                        if self.grid[y][x] != '' and self.grid[y][x] != word[i]:
                            can_place = False
                            break
                    
                    if can_place:
                        for i in range(len(word)):
                            x, y = x_start + i * dx, y_start + i * dy
                            self.grid[y][x] = word[i]
                        self.word_locations[word] = ((x_start, y_start), (x_end, y_end))
                        placed = True

        # Fill empty cells with random letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == '':
                    self.grid[y][x] = self.np_random.choice(list(alphabet))

    def step(self, action):
        reward = -0.01  # Small penalty for taking a step to encourage speed
        self.steps += 1
        
        self._handle_input(action)
        reward += self._process_selection(action)
        self._update_effects()

        terminated = self._check_termination()
        if terminated:
            if len(self.found_words) == self.WORDS_TO_FIND:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty

        self.previous_action = action

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, _, _ = action
        # Move cursor
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        # Wrap cursor around grid
        self.cursor_pos[0] %= self.GRID_SIZE
        self.cursor_pos[1] %= self.GRID_SIZE

    def _process_selection(self, action):
        reward = 0
        _, space_held, shift_held = action
        prev_space_held = self.previous_action[1] == 1
        prev_shift_held = self.previous_action[2] == 1

        space_pressed = space_held and not prev_space_held
        shift_pressed = shift_held and not prev_shift_held

        if shift_pressed and self.selection_start is not None:
            # Cancel selection
            self.selection_start = None
            # sound: cancel_sfx
            
        elif space_pressed:
            if self.selection_start is None:
                # Start selection
                self.selection_start = tuple(self.cursor_pos)
                # sound: click_sfx
            else:
                # Finalize selection
                start_x, start_y = self.selection_start
                end_x, end_y = self.cursor_pos

                # Extract word from grid
                selected_word = ""
                dx = np.sign(end_x - start_x)
                dy = np.sign(end_y - start_y)

                # Check for valid line (horizontal, vertical, or diagonal)
                if (start_x == end_x or start_y == end_y or
                    abs(end_x - start_x) == abs(end_y - start_y)):
                    
                    curr_x, curr_y = start_x, start_y
                    while True:
                        selected_word += self.grid[curr_y][curr_x]
                        if (curr_x, curr_y) == (end_x, end_y):
                            break
                        curr_x += dx
                        curr_y += dy
                
                # Check if word is correct
                if selected_word in self.word_list and selected_word not in self.found_words:
                    self.found_words.add(selected_word)
                    self.score += 10
                    reward += 10
                    self.visual_effects.append(("success", self.selection_start, self.cursor_pos, 15))
                    # sound: success_sfx
                elif selected_word[::-1] in self.word_list and selected_word[::-1] not in self.found_words:
                    self.found_words.add(selected_word[::-1])
                    self.score += 10
                    reward += 10
                    self.visual_effects.append(("success", self.selection_start, self.cursor_pos, 15))
                    # sound: success_sfx
                else:
                    reward -= 2
                    self.visual_effects.append(("fail", self.selection_start, self.cursor_pos, 15))
                    # sound: fail_sfx

                self.selection_start = None
        return reward

    def _check_termination(self):
        if len(self.found_words) == self.WORDS_TO_FIND:
            return True
        if self.steps >= self.MAX_TIME_STEPS:
            return True
        return False

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
            "time_left": self.MAX_TIME_STEPS - self.steps,
            "words_found": len(self.found_words),
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (0, i * self.CELL_SIZE), (self.GRID_WIDTH, i * self.CELL_SIZE))

        # Draw letters
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                letter_surf = self.font_letter.render(self.grid[y][x], True, self.COLOR_LETTER)
                letter_rect = letter_surf.get_rect(center=(x * self.CELL_SIZE + self.CELL_SIZE // 2, y * self.CELL_SIZE + self.CELL_SIZE // 2))
                self.screen.blit(letter_surf, letter_rect)

        # Draw visual effects (flashes)
        for effect in self.visual_effects:
            if effect[0] in ["success", "fail"]:
                color = self.COLOR_SUCCESS_FLASH if effect[0] == "success" else self.COLOR_FAIL_FLASH
                alpha = int(255 * (effect[3] / 15))
                self._draw_selection_line(effect[1], effect[2], color, alpha)

        # Draw active selection line
        if self.selection_start:
            self._draw_selection_line(self.selection_start, self.cursor_pos, self.COLOR_SELECTION[:3], self.COLOR_SELECTION[3])

        # Draw cursor
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, cursor_rect.topleft)

    def _draw_selection_line(self, start_pos, end_pos, color, alpha):
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        dx = np.sign(end_x - start_x)
        dy = np.sign(end_y - start_y)

        if not (start_x == end_x or start_y == end_y or abs(end_x - start_x) == abs(end_y - start_y)):
            return # Not a valid line

        curr_x, curr_y = start_x, start_y
        while True:
            rect = pygame.Rect(curr_x * self.CELL_SIZE, curr_y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill((*color, alpha))
            self.screen.blit(s, rect.topleft)
            if (curr_x, curr_y) == (end_x, end_y):
                break
            curr_x += dx
            curr_y += dy

    def _render_ui(self):
        ui_x_start = self.GRID_WIDTH + 20
        
        # Timer
        time_left_sec = (self.MAX_TIME_STEPS - self.steps) / 30 # Rough estimate for display
        timer_text = f"TIME: {max(0, int(time_left_sec // 60)):02}:{max(0, int(time_left_sec % 60)):02}"
        timer_surf = self.font_ui_title.render(timer_text, True, self.COLOR_TEXT)
        self.screen.blit(timer_surf, (ui_x_start, 20))

        # Score
        score_text = f"FOUND: {len(self.found_words)}/{self.WORDS_TO_FIND}"
        score_surf = self.font_ui_title.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (ui_x_start, 50))

        # Word List
        word_list_y = 90
        for word in sorted(self.word_list):
            if word in self.found_words:
                word_surf = self.font_ui_strikethrough.render(word, True, self.COLOR_WORD_FOUND)
            else:
                word_surf = self.font_ui_word.render(word, True, self.COLOR_TEXT)
            self.screen.blit(word_surf, (ui_x_start, word_list_y))
            word_list_y += 20

    def _update_effects(self):
        self.visual_effects = [(t, p1, p2, timer - 1) for t, p1, p2, timer in self.visual_effects if timer > 0]

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation:
        """
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

# Example of how to run the environment
if __name__ == '__main__':
    import time

    # To run with human interaction
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a different window for rendering the game
    pygame.display.set_caption("Word Search")
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    terminated = False
    
    # Map Pygame keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP:    (1, 0, 0),
        pygame.K_DOWN:  (2, 0, 0),
        pygame.K_LEFT:  (3, 0, 0),
        pygame.K_RIGHT: (4, 0, 0),
    }

    action = np.array([0, 0, 0]) # No-op, released, released
    
    while not terminated:
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
        keys = pygame.key.get_pressed()
        
        # Reset movement part of action
        action[0] = 0
        
        # Movement (only one direction at a time)
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4

        # Buttons
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if info['steps'] % 10 == 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Words Found: {info['words_found']}")

        # Control the frame rate
        env.clock.tick(30)

    print("Game Over!")
    print(f"Final Score: {info['score']}, Words Found: {info['words_found']}/{env.WORDS_TO_FIND}")
    time.sleep(3)
    env.close()