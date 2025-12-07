
# Generated: 2025-08-28T01:03:35.890657
# Source Brief: brief_03992.md
# Brief Index: 3992

        
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
        "Controls: Arrow keys to move cursor. Space to select start/end of a word. Shift to clear selection."
    )

    game_description = (
        "Find hidden words in a grid of letters before the timer runs out. Select the first and last letter of a word to solve it."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 16
    GRID_ROWS = 10
    MAX_STEPS = 3600  # 120 seconds at 30 FPS
    FPS = 30

    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (0, 200, 255)
    COLOR_SELECTION = (255, 200, 0)
    COLOR_FOUND = (50, 220, 120)
    COLOR_MISS = (255, 80, 80)
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_TIMER_WARN = (255, 100, 0)

    WORD_LIST = [
        "PYTHON", "GYMNASIUM", "AGENT", "REWARD", "ACTION", "STATE", "POLICY",
        "LEARNING", "NEURAL", "DEEP", "GRID", "SEARCH", "PUZZLE", "VECTOR", "TENSOR"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_grid = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_ui = pygame.font.SysFont("sans-serif", 24, bold=True)
        self.font_feedback = pygame.font.SysFont("sans-serif", 32, bold=True)
        
        # Grid layout calculation
        self.grid_area_height = 320
        self.cell_size = min((self.SCREEN_WIDTH - 40) // self.GRID_COLS, self.grid_area_height // self.GRID_ROWS)
        self.grid_width = self.cell_size * self.GRID_COLS
        self.grid_height = self.cell_size * self.GRID_ROWS
        self.grid_x_offset = (self.SCREEN_WIDTH - self.grid_width) // 2
        self.grid_y_offset = 80 + (self.grid_area_height - self.grid_height) // 2

        # State variables (initialized in reset)
        self.grid = None
        self.solutions = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        self.cursor_pos = None
        self.selection_start = None
        self.found_words = None
        self.prev_space_held = False
        self.feedback_effect = None # tuple: (type, duration) e.g. ("found", 15)

        self.reset()
        
        # self.validate_implementation() # Optional: Call to self-check

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.solutions = []
        
        words_to_place = random.sample(self.WORD_LIST, len(self.WORD_LIST))

        for word in words_to_place:
            word = word.upper()
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                attempts += 1
                direction = random.choice([(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)])
                dx, dy = direction

                if dx != 0:
                    start_x = random.randint(0, self.GRID_COLS - len(word))
                else:
                    start_x = random.randint(0, self.GRID_COLS - 1)

                if dy != 0:
                    start_y = random.randint(0, self.GRID_ROWS - len(word))
                else:
                    start_y = random.randint(0, self.GRID_ROWS - 1)

                end_x = start_x + (len(word) - 1) * dx
                end_y = start_y + (len(word) - 1) * dy

                if not (0 <= start_x < self.GRID_COLS and 0 <= end_x < self.GRID_COLS and
                        0 <= start_y < self.GRID_ROWS and 0 <= end_y < self.GRID_ROWS):
                    continue

                can_place = True
                for i in range(len(word)):
                    x, y = start_x + i * dx, start_y + i * dy
                    if self.grid[y][x] != '' and self.grid[y][x] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i in range(len(word)):
                        x, y = start_x + i * dx, start_y + i * dy
                        self.grid[y][x] = word[i]
                    
                    self.solutions.append({
                        "word": word,
                        "start": (start_x, start_y),
                        "end": (end_x, end_y)
                    })
                    placed = True

        # Fill empty cells with random letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == '':
                    self.grid[r][c] = random.choice(string.ascii_uppercase)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_grid()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selection_start = None
        self.found_words = set()
        self.prev_space_held = False
        self.feedback_effect = None

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.prev_space_held
        self.prev_space_held = space_held
        
        reward = 0
        self.steps += 1
        self.time_remaining -= 1

        # Handle feedback effect duration
        if self.feedback_effect:
            effect_type, duration = self.feedback_effect
            if duration > 0:
                self.feedback_effect = (effect_type, duration - 1)
            else:
                self.feedback_effect = None

        # 1. Handle actions
        # Movement
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1) # Up
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1) # Down
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1) # Left
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1) # Right

        # Reset selection
        if shift_held:
            if self.selection_start:
                # Small penalty for canceling a selection
                reward -= 0.05
                self.selection_start = None
        
        # Make selection
        elif space_pressed:
            cursor_tuple = tuple(self.cursor_pos)
            if self.selection_start is None:
                self.selection_start = cursor_tuple
                # Reward for selecting a letter that is part of an unfound word
                in_word = False
                for sol in self.solutions:
                    if sol["word"] in self.found_words: continue
                    # Check if cursor is on the path of an unfound word
                    min_x, max_x = sorted((sol["start"][0], sol["end"][0]))
                    min_y, max_y = sorted((sol["start"][1], sol["end"][1]))
                    if min_x <= cursor_tuple[0] <= max_x and min_y <= cursor_tuple[1] <= max_y:
                         in_word = True
                         break
                reward += 0.1 if in_word else -0.1

            else:
                # Second selection, check for a word
                is_new_word, found_word_obj = self._check_word(self.selection_start, cursor_tuple)
                if is_new_word:
                    # Sound: Word Found!
                    reward += 10
                    self.score += 1
                    self.found_words.add(found_word_obj["word"])
                    self.feedback_effect = ("found", self.FPS // 2)
                else:
                    # Sound: Incorrect
                    reward -= 0.5
                    self.feedback_effect = ("miss", self.FPS // 3)
                self.selection_start = None

        # 2. Check for termination
        terminated = False
        if self.time_remaining <= 0:
            reward -= 50
            terminated = True
            self.feedback_effect = ("miss", self.FPS)
        elif len(self.found_words) == len(self.solutions):
            reward += 50
            terminated = True
            self.feedback_effect = ("win", self.FPS * 2)
        elif self.steps >= self.MAX_STEPS:
            terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_word(self, start_pos, end_pos):
        for sol in self.solutions:
            if sol["word"] in self.found_words:
                continue
            
            is_match = (sol["start"] == start_pos and sol["end"] == end_pos) or \
                       (sol["start"] == end_pos and sol["end"] == start_pos)
            
            if is_match:
                return True, sol
        return False, None

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_x_offset, self.grid_y_offset, self.grid_width, self.grid_height)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=8)

        # Draw found words highlights
        for sol in self.solutions:
            if sol["word"] in self.found_words:
                self._draw_word_highlight(sol, self.COLOR_FOUND, 6)

        # Draw current selection highlight
        if self.selection_start:
            start_cx = self.grid_x_offset + self.selection_start[0] * self.cell_size + self.cell_size / 2
            start_cy = self.grid_y_offset + self.selection_start[1] * self.cell_size + self.cell_size / 2
            cursor_cx = self.grid_x_offset + self.cursor_pos[0] * self.cell_size + self.cell_size / 2
            cursor_cy = self.grid_y_offset + self.cursor_pos[1] * self.cell_size + self.cell_size / 2
            
            pygame.draw.line(self.screen, self.COLOR_SELECTION, (start_cx, start_cy), (cursor_cx, cursor_cy), 6)
            pygame.gfxdraw.aacircle(self.screen, int(start_cx), int(start_cy), self.cell_size // 3, self.COLOR_SELECTION)
            pygame.gfxdraw.filled_circle(self.screen, int(start_cx), int(start_cy), self.cell_size // 3, self.COLOR_SELECTION)


        # Draw letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                letter = self.grid[r][c]
                text_surf = self.font_grid.render(letter, True, self.COLOR_TEXT)
                text_rect = text_surf.get_rect(center=(
                    self.grid_x_offset + c * self.cell_size + self.cell_size / 2,
                    self.grid_y_offset + r * self.cell_size + self.cell_size / 2
                ))
                self.screen.blit(text_surf, text_rect)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.grid_x_offset + self.cursor_pos[0] * self.cell_size,
            self.grid_y_offset + self.cursor_pos[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
    
    def _draw_word_highlight(self, sol, color, width):
        start_pos_pixel = (
            self.grid_x_offset + sol["start"][0] * self.cell_size + self.cell_size / 2,
            self.grid_y_offset + sol["start"][1] * self.cell_size + self.cell_size / 2
        )
        end_pos_pixel = (
            self.grid_x_offset + sol["end"][0] * self.cell_size + self.cell_size / 2,
            self.grid_y_offset + sol["end"][1] * self.cell_size + self.cell_size / 2
        )
        pygame.draw.line(self.screen, color, start_pos_pixel, end_pos_pixel, width=self.cell_size)
        
        # Re-render letters on top of the highlight for clarity
        dx = np.sign(sol["end"][0] - sol["start"][0])
        dy = np.sign(sol["end"][1] - sol["start"][1])
        for i in range(len(sol["word"])):
            c = sol["start"][0] + i * dx
            r = sol["start"][1] + i * dy
            letter = self.grid[r][c]
            text_surf = self.font_grid.render(letter, True, self.COLOR_GRID_BG)
            text_rect = text_surf.get_rect(center=(
                self.grid_x_offset + c * self.cell_size + self.cell_size / 2,
                self.grid_y_offset + r * self.cell_size + self.cell_size / 2
            ))
            self.screen.blit(text_surf, text_rect)


    def _render_ui(self):
        # Draw score
        score_text = f"FOUND: {self.score} / {len(self.solutions)}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 20))

        # Draw timer
        time_seconds = self.time_remaining / self.FPS
        minutes = int(time_seconds) // 60
        seconds = int(time_seconds) % 60
        timer_text = f"TIME: {minutes:02}:{seconds:02}"
        timer_color = self.COLOR_TIMER_WARN if time_seconds < 20 and self.time_remaining % self.FPS > self.FPS/2 else self.COLOR_UI_TEXT
        timer_surf = self.font_ui.render(timer_text, True, timer_color)
        timer_rect = timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(timer_surf, timer_rect)

        # Draw feedback effects
        if self.feedback_effect:
            effect_type, _ = self.feedback_effect
            if effect_type == "miss":
                s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
                s.fill((self.COLOR_MISS[0], self.COLOR_MISS[1], self.COLOR_MISS[2], 100))
                self.screen.blit(s, (0, 0))
            elif effect_type == "win":
                win_text = self.font_feedback.render("ALL WORDS FOUND!", True, self.COLOR_FOUND)
                win_rect = win_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
                self.screen.blit(win_text, win_rect)
            elif self.game_over and self.time_remaining <= 0:
                 lose_text = self.font_feedback.render("TIME UP!", True, self.COLOR_MISS)
                 lose_rect = lose_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
                 self.screen.blit(lose_text, lose_rect)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_found": len(self.found_words),
            "time_remaining_steps": self.time_remaining
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

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Setup a window to display the game
    pygame.display.set_caption("Word Search")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = env.action_space.sample()
    action.fill(0) # Start with no-op

    while not terminated:
        # --- Human input mapping ---
        movement, space_held, shift_held = 0, 0, 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = np.array([movement, space_held, shift_held])
        # --- End human input ---

        obs, reward, terminated, truncated, info = env.step(action)

        # --- Pygame event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        # Update the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
            # Wait a bit before closing
            pygame.time.wait(3000)

    env.close()