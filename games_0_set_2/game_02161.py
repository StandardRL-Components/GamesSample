
# Generated: 2025-08-28T04:00:51.273469
# Source Brief: brief_02161.md
# Brief Index: 2161

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select/deselect letters. Press shift to submit your word."
    )

    game_description = (
        "Find hidden words in a procedurally generated grid of letters before the time runs out. Correctly found words are highlighted in green."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 15
    NUM_WORDS = 10
    MAX_TIME_SECONDS = 60
    FPS = 30 # For visual timing and step-based timer
    MAX_STEPS = MAX_TIME_SECONDS * FPS

    # --- Colors ---
    COLOR_BG = (250, 250, 250)
    COLOR_GRID_LINES = (210, 210, 210)
    COLOR_LETTER = (50, 50, 50)
    COLOR_CURSOR = (0, 122, 255)
    COLOR_SELECT_FILL = (0, 122, 255, 60)
    COLOR_FOUND_FILL = (40, 205, 65, 80)
    COLOR_FLASH_BAD = (255, 59, 48, 150)
    COLOR_UI_TEXT = (20, 20, 20)
    COLOR_UI_FOUND = (150, 150, 150)
    
    WORD_BANK = [
        "PYTHON", "AGENT", "REWARD", "STATE", "ACTION", "POLICY", "GRID", "LEARN",
        "SOLVE", "SEARCH", "PUZZLE", "GYM", "SPACE", "VECTOR", "TENSOR", "MODEL",
        "GAME", "CODE", "PLAY", "TEST", "VISUAL", "KERNEL", "LAYER", "FRAME", "PIXEL"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_letter = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_ui = pygame.font.SysFont("Segoe UI", 20)
            self.font_ui_title = pygame.font.SysFont("Segoe UI", 22, bold=True)
            self.font_ui_found = pygame.font.SysFont("Segoe UI", 20)
            self.font_ui_found.set_strikethrough(True)
        except pygame.error:
            # Fallback to default font if system fonts are not available
            self.font_letter = pygame.font.Font(None, 22)
            self.font_ui = pygame.font.Font(None, 24)
            self.font_ui_title = pygame.font.Font(None, 26)
            self.font_ui_found = pygame.font.Font(None, 24)
            self.font_ui_found.set_strikethrough(True)

        # Game state variables
        self.grid = []
        self.word_data = {}
        self.target_words = []
        self.found_words = set()
        self.cursor_pos = [0, 0]
        self.selected_cells = []
        self.found_cells = set()

        self.steps_remaining = 0
        self.score = 0
        self.game_over = False
        self.steps = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.feedback_flash_timer = 0
        self.feedback_flash_color = (0,0,0,0)

        self.grid_top_left = (220, 40)
        self.cell_size = 24
        self.grid_pixel_size = self.GRID_SIZE * self.cell_size
        
        self.np_random = None

        # This will be called last after all attributes are defined
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.np_random is None: # Initialize RNG on first reset
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.steps_remaining = self.MAX_STEPS
        
        self.found_words = set()
        self.selected_cells = []
        self.found_cells = set()
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        
        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True

        self.feedback_flash_timer = 0

        # Generate a valid grid, retrying if the algorithm fails
        generation_success = False
        while not generation_success:
            generation_success = self._generate_grid()

        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.target_words = sorted([w for w in self.np_random.choice(self.WORD_BANK, self.NUM_WORDS, replace=False)])
        self.word_data = {}

        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]

        for word in self.target_words:
            placed = False
            for _ in range(100): # 100 attempts to place a word
                word_to_place = word if self.np_random.random() < 0.5 else word[::-1]
                d_r, d_c = directions[self.np_random.integers(0, len(directions))]
                
                start_r = self.np_random.integers(0, self.GRID_SIZE)
                start_c = self.np_random.integers(0, self.GRID_SIZE)
                
                end_r = start_r + (len(word_to_place) - 1) * d_r
                end_c = start_c + (len(word_to_place) - 1) * d_c

                if 0 <= end_r < self.GRID_SIZE and 0 <= end_c < self.GRID_SIZE:
                    can_place = True
                    cells_to_occupy = []
                    for i in range(len(word_to_place)):
                        r, c = start_r + i * d_r, start_c + i * d_c
                        cells_to_occupy.append((r, c))
                        if self.grid[r][c] != '' and self.grid[r][c] != word_to_place[i]:
                            can_place = False
                            break
                    
                    if can_place:
                        for i, char in enumerate(word_to_place):
                            r, c = start_r + i * d_r, start_c + i * d_c
                            self.grid[r][c] = char
                        self.word_data[word] = cells_to_occupy
                        placed = True
                        break
            if not placed:
                return False # Generation failed, reset will try again

        # Fill empty cells
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r][c] == '':
                    self.grid[r][c] = self.np_random.choice(list(alphabet))
        
        return True

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        reward = 0
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        space_press = space_held and not self.prev_space_held
        shift_press = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        if movement != 0: self._handle_movement(movement)
        if space_press: reward += self._handle_selection()
            
        if shift_press:
            submission_reward, correct = self._handle_submission()
            reward += submission_reward
            if not correct and len(self.selected_cells) > 0:
                self.steps_remaining = max(0, self.steps_remaining - (1 * self.FPS)) # Time penalty
                self.feedback_flash_color = self.COLOR_FLASH_BAD
                self.feedback_flash_timer = 5
            self.selected_cells = []

        self.steps += 1
        self.steps_remaining -= 1
        if self.feedback_flash_timer > 0: self.feedback_flash_timer -= 1
        
        terminated = self._check_termination()
        if terminated:
            self.game_over = True
            if len(self.found_words) == self.NUM_WORDS:
                reward += 100 # Win
            else:
                reward -= 50 # Lose

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        r, c = self.cursor_pos
        if movement == 1: r -= 1 # Up
        elif movement == 2: r += 1 # Down
        elif movement == 3: c -= 1 # Left
        elif movement == 4: c += 1 # Right
        self.cursor_pos = [r % self.GRID_SIZE, c % self.GRID_SIZE]

    def _handle_selection(self):
        cell = tuple(self.cursor_pos)
        if cell in self.found_cells: return -0.1
        
        if cell in self.selected_cells: self.selected_cells.remove(cell)
        else: self.selected_cells.append(cell)
        
        is_useful = any(word not in self.found_words and cell in path for word, path in self.word_data.items())
        return 0.1 if is_useful else -0.1

    def _handle_submission(self):
        if not self.selected_cells: return 0, False

        submitted_cell_set = set(self.selected_cells)
        for word, path in self.word_data.items():
            if word not in self.found_words and submitted_cell_set == set(path):
                self.found_words.add(word)
                self.found_cells.update(path)
                self.score += 100
                return 10.0, True # Event-based reward for finding a word
        return 0, False

    def _check_termination(self):
        return self.steps_remaining <= 0 or len(self.found_words) == self.NUM_WORDS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_ui()
        self._render_game()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        gx, gy = self.grid_top_left
        s = self.cell_size
        gs = self.grid_pixel_size

        s_select = pygame.Surface((s, s), pygame.SRCALPHA); s_select.fill(self.COLOR_SELECT_FILL)
        s_found = pygame.Surface((s, s), pygame.SRCALPHA); s_found.fill(self.COLOR_FOUND_FILL)

        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(gx + c * s, gy + r * s, s, s)
                
                if (r, c) in self.found_cells: self.screen.blit(s_found, cell_rect.topleft)
                elif (r, c) in self.selected_cells: self.screen.blit(s_select, cell_rect.topleft)

                letter_surf = self.font_letter.render(self.grid[r][c], True, self.COLOR_LETTER)
                self.screen.blit(letter_surf, letter_surf.get_rect(center=cell_rect.center))
        
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (gx + i * s, gy), (gx + i * s, gy + gs))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (gx, gy + i * s), (gx + gs, gy + i * s))

        cr, cc = self.cursor_pos
        cursor_rect = pygame.Rect(gx + cc * s, gy + cr * s, s, s)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=2)

        if self.feedback_flash_timer > 0:
            flash_surface = pygame.Surface((gs, gs), pygame.SRCALPHA)
            flash_surface.fill(self.feedback_flash_color)
            self.screen.blit(flash_surface, (gx, gy))

    def _render_ui(self):
        title_surf = self.font_ui_title.render("WORDS TO FIND", True, self.COLOR_UI_TEXT)
        self.screen.blit(title_surf, (20, 20))
        
        for i, word in enumerate(self.target_words):
            is_found = word in self.found_words
            font = self.font_ui_found if is_found else self.font_ui
            color = self.COLOR_UI_FOUND if is_found else self.COLOR_UI_TEXT
            word_surf = font.render(word, True, color)
            self.screen.blit(word_surf, (20, 60 + i * 25))

        time_left = max(0, self.steps_remaining / self.FPS)
        timer_text = f"TIME: {time_left:.1f}"
        timer_surf = self.font_ui_title.render(timer_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_surf, timer_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 10)))

        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui_title.render(score_text, True, self.COLOR_UI_TEXT)
        score_rect_center = (self.grid_top_left[0] + self.grid_pixel_size / 2, 25)
        self.screen.blit(score_surf, score_surf.get_rect(center=score_rect_center))

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls.
    # Set `SDL_VIDEODRIVER=dummy` in your environment for headless execution.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Search")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        action = [movement, keys[pygame.K_SPACE], keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    env.close()