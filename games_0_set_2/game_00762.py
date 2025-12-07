
# Generated: 2025-08-27T14:41:26.129411
# Source Brief: brief_00762.md
# Brief Index: 762

        
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
        "Controls: Arrow keys to move cursor. Space to select the first letter of a word, Shift to select the last letter and submit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find all 10 hidden words in the grid before the timer runs out. Correct words grant points, but wrong submissions cost you time."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.width, self.height = 640, 400
        self.grid_size = 10
        self.cell_size = 32
        self.grid_offset_x = (self.width - self.grid_size * self.cell_size) // 2
        self.grid_offset_y = 20

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 16)
        self.font_small = pygame.font.SysFont("Consolas", 12)

        # Colors
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID_BG = (30, 35, 40)
        self.COLOR_GRID_LINE = (50, 55, 60)
        self.COLOR_LETTER = (220, 220, 220)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_START_SELECT = (255, 200, 0, 100)
        self.COLOR_FOUND_WORD_BG = (0, 150, 255, 100)
        self.COLOR_FOUND_WORD_LETTER = (100, 200, 255)
        self.COLOR_UI_TEXT = (200, 200, 200)
        self.COLOR_SUCCESS = (0, 255, 100)
        self.COLOR_FAIL = (255, 50, 50)
        self.COLOR_TIMER_BAR = (0, 150, 255)
        self.COLOR_TIMER_BAR_BG = (50, 55, 60)

        # Word list
        self.WORD_LIST = ["PYTHON", "GYMNASIUM", "RENDER", "ACTION", "REWARD", "AGENT", "POLICY", "STATE", "RESET", "STEP"]

        # Game state variables are initialized in reset()
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.max_steps = 1800 # 60 seconds * 30fps equivalent
        self.steps_remaining = self.max_steps

        self.cursor_pos = [0, 0]
        self.start_selection = None
        self.found_words = set()
        self.word_locations = {}
        self.word_coords = {}

        self.feedback_flash = 0 # Countdown timer for flash effect
        self.feedback_color = self.COLOR_SUCCESS

        self._generate_grid()

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.word_locations.clear()
        self.word_coords.clear()

        for word in self.WORD_LIST:
            placed = False
            for _ in range(100): # Max 100 attempts to place a word
                direction = self.np_random.choice(['h', 'v', 'd1', 'd2'])
                if direction == 'h': # Horizontal
                    row, col = self.np_random.integers(0, self.grid_size), self.np_random.integers(0, self.grid_size - len(word))
                    dx, dy = 1, 0
                elif direction == 'v': # Vertical
                    row, col = self.np_random.integers(0, self.grid_size - len(word)), self.np_random.integers(0, self.grid_size)
                    dx, dy = 0, 1
                elif direction == 'd1': # Diagonal \
                    row, col = self.np_random.integers(0, self.grid_size - len(word)), self.np_random.integers(0, self.grid_size - len(word))
                    dx, dy = 1, 1
                else: # Diagonal /
                    row, col = self.np_random.integers(len(word) - 1, self.grid_size), self.np_random.integers(0, self.grid_size - len(word))
                    dx, dy = 1, -1

                # Check for conflicts
                can_place = True
                for i in range(len(word)):
                    r, c = row + i * dy, col + i * dx
                    if self.grid[r][c] != '' and self.grid[r][c] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    start_pos = (col, row)
                    end_pos = (col + (len(word) - 1) * dx, row + (len(word) - 1) * dy)
                    self.word_locations[word] = (start_pos, end_pos)
                    
                    coords = set()
                    for i in range(len(word)):
                        r, c = row + i * dy, col + i * dx
                        self.grid[r][c] = word[i]
                        coords.add((c, r))
                    self.word_coords[word] = coords
                    placed = True
                    break
            if not placed:
                # Failsafe: if a word can't be placed, restart the whole generation
                return self._generate_grid()
        
        # Fill empty cells with random letters
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c] == '':
                    self.grid[r][c] = chr(self.np_random.integers(65, 91))

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle player actions ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.grid_size - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.grid_size - 1, self.cursor_pos[0] + 1)

        word_found_event = False
        wrong_submission_event = False

        if space_held:
            # sound: select_start_letter.wav
            self.start_selection = tuple(self.cursor_pos)
        
        if shift_held and self.start_selection is not None:
            end_selection = tuple(self.cursor_pos)
            
            # Prevent submitting a single cell
            if self.start_selection == end_selection:
                self.start_selection = None
            else:
                found_match = False
                for word, (start, end) in self.word_locations.items():
                    if word not in self.found_words:
                        if (self.start_selection == start and end_selection == end) or \
                           (self.start_selection == end and end_selection == start):
                            self.found_words.add(word)
                            self.score += 1
                            word_found_event = True
                            found_match = True
                            # sound: success.wav
                            self.feedback_flash = 15
                            self.feedback_color = self.COLOR_SUCCESS
                            break
                
                if not found_match:
                    wrong_submission_event = True
                    # sound: failure.wav
                    self.feedback_flash = 15
                    self.feedback_color = self.COLOR_FAIL

                self.start_selection = None

        # --- Update game state ---
        self.steps += 1
        self.steps_remaining -= 1
        if wrong_submission_event:
            self.steps_remaining -= 15 # 0.5 sec penalty

        reward = self._calculate_reward(word_found_event)
        
        terminated = (len(self.found_words) == len(self.WORD_LIST)) or (self.steps_remaining <= 0)
        if terminated and len(self.found_words) == len(self.WORD_LIST):
            reward += 50 # Goal-oriented reward

        if self.feedback_flash > 0:
            self.feedback_flash -= 1
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_reward(self, word_found_event):
        # Base reward for exploring
        reward = -0.01
        
        # Check if cursor is on a letter of an unfound word
        cursor_tuple = tuple(self.cursor_pos)
        for word, coords in self.word_coords.items():
            if word not in self.found_words and cursor_tuple in coords:
                reward = 0.01
                break
        
        if word_found_event:
            reward += 10
            
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.grid_offset_x, self.grid_offset_y, self.grid_size * self.cell_size, self.grid_size * self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw highlights for found words
        for word in self.found_words:
            start_pos, end_pos = self.word_locations[word]
            start_px = (self.grid_offset_x + start_pos[0] * self.cell_size + self.cell_size // 2, self.grid_offset_y + start_pos[1] * self.cell_size + self.cell_size // 2)
            end_px = (self.grid_offset_x + end_pos[0] * self.cell_size + self.cell_size // 2, self.grid_offset_y + end_pos[1] * self.cell_size + self.cell_size // 2)
            pygame.draw.line(self.screen, self.COLOR_FOUND_WORD_BG, start_px, end_px, self.cell_size - 4)

        # Draw grid lines
        for i in range(self.grid_size + 1):
            x = self.grid_offset_x + i * self.cell_size
            y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.grid_offset_y), (x, self.grid_offset_y + self.grid_size * self.cell_size))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.grid_offset_x, y), (self.grid_offset_x + self.grid_size * self.cell_size, y))

        # Draw letters
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                letter = self.grid[r][c]
                is_on_found_word = False
                for word in self.found_words:
                    if (c,r) in self.word_coords[word]:
                        is_on_found_word = True
                        break
                
                color = self.COLOR_FOUND_WORD_LETTER if is_on_found_word else self.COLOR_LETTER
                text_surf = self.font_large.render(letter, True, color)
                text_rect = text_surf.get_rect(center=(self.grid_offset_x + c * self.cell_size + self.cell_size // 2, self.grid_offset_y + r * self.cell_size + self.cell_size // 2))
                self.screen.blit(text_surf, text_rect)
        
        # Draw start selection highlight
        if self.start_selection is not None:
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            s.fill(self.COLOR_START_SELECT)
            self.screen.blit(s, (self.grid_offset_x + self.start_selection[0] * self.cell_size, self.grid_offset_y + self.start_selection[1] * self.cell_size))
            
            # Draw line to current cursor
            start_px = (self.grid_offset_x + self.start_selection[0] * self.cell_size + self.cell_size // 2, self.grid_offset_y + self.start_selection[1] * self.cell_size + self.cell_size // 2)
            cursor_px = (self.grid_offset_x + self.cursor_pos[0] * self.cell_size + self.cell_size // 2, self.grid_offset_y + self.cursor_pos[1] * self.cell_size + self.cell_size // 2)
            pygame.draw.line(self.screen, self.COLOR_CURSOR, start_px, cursor_px, 2)


        # Draw cursor
        cursor_rect = pygame.Rect(self.grid_offset_x + self.cursor_pos[0] * self.cell_size, self.grid_offset_y + self.cursor_pos[1] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3)

        # Draw feedback flash
        if self.feedback_flash > 0:
            flash_alpha = int(100 * (self.feedback_flash / 15))
            flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            flash_surface.fill((*self.feedback_color, flash_alpha))
            self.screen.blit(flash_surface, (0, 0))

    def _render_ui(self):
        # Draw Score
        score_text = self.font_medium.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Draw Timer Bar
        timer_bar_width = 200
        timer_bar_height = 15
        timer_ratio = max(0, self.steps_remaining / self.max_steps)
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR_BG, (self.width - timer_bar_width - 10, 10, timer_bar_width, timer_bar_height))
        pygame.draw.rect(self.screen, self.COLOR_TIMER_BAR, (self.width - timer_bar_width - 10, 10, timer_bar_width * timer_ratio, timer_bar_height))

        # Draw Word List
        word_list_y_start = self.grid_offset_y + self.grid_size * self.cell_size + 10
        col_width = 120
        for i, word in enumerate(self.WORD_LIST):
            col = i // 5
            row = i % 5
            is_found = word in self.found_words
            color = self.COLOR_SUCCESS if is_found else self.COLOR_UI_TEXT
            text_surf = self.font_small.render(word, True, color)
            if is_found:
                pygame.draw.line(self.screen, self.COLOR_SUCCESS, (10 + col * col_width, word_list_y_start + row * 15 + 6), (10 + col * col_width + text_surf.get_width(), word_list_y_start + row * 15 + 6))
            self.screen.blit(text_surf, (10 + col * col_width, word_list_y_start + row * 15))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "steps_remaining": self.steps_remaining,
            "found_words": len(self.found_words),
            "words_to_find": len(self.WORD_LIST),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # pip install gymnasium[classic-control]
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Word Search Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # Game loop for human play
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000)
            
        clock.tick(10) # Control human play speed
        
    env.close()