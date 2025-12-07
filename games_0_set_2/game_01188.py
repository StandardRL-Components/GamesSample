
# Generated: 2025-08-27T16:19:38.896567
# Source Brief: brief_01188.md
# Brief Index: 1188

        
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
        "Controls: ↑↓←→ to move cursor. Space to set word start/end. Shift to submit/cancel."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find hidden words in a procedurally generated grid before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    GRID_SIZE = 15
    CELL_SIZE = 24
    GRID_WIDTH = GRID_SIZE * CELL_SIZE
    GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    
    UI_WIDTH = 280 # Increased width for better layout
    SCREEN_WIDTH = GRID_WIDTH + UI_WIDTH
    SCREEN_HEIGHT = GRID_HEIGHT

    MAX_STEPS = 30 * 120 # 120 seconds at 30 FPS
    NUM_WORDS = 10

    # --- Colors ---
    COLOR_BG = (15, 25, 40)
    COLOR_GRID_LINE = (30, 45, 65)
    COLOR_LETTER = (180, 190, 210)
    
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION_START = (0, 150, 255)
    COLOR_SELECTION_PATH = (0, 150, 255)
    
    COLOR_FOUND_WORD_BG = (40, 180, 120)
    COLOR_FOUND_WORD_LETTER = (255, 255, 255)

    COLOR_UI_BG = (25, 35, 55)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_WORD_FOUND = (80, 220, 160)
    COLOR_UI_TIMER = (255, 80, 80)
    
    COLOR_SUCCESS_FLASH = (80, 220, 160)
    COLOR_FAIL_FLASH = (220, 80, 80)

    # --- Word Bank ---
    WORD_BANK = [
        "PYTHON", "GYMNASIUM", "REWARD", "ACTION", "AGENT", "POLICY", "STATE", 
        "EPISODE", "LEARNING", "VECTOR", "TENSOR", "KERAS", "TORCH", "MODEL",
        "NEURAL", "NETWORK", "SPACE", "RESET", "STEP", "RENDER", "ALGORITHM",
        "SEARCH", "GRID", "PUZZLE", "SOLVE", "TIMER", "SCORE", "VISUAL", "CODE"
    ]


    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # NOTE: The brief specifies a 640x400 screen.
        # This implementation uses a dynamic size based on grid parameters for better design.
        # To strictly match the brief, GRID_SIZE, CELL_SIZE, etc. would need to be
        # adjusted to produce a 640x400 total.
        # For example: GRID_SIZE=15, CELL_SIZE=26, UI_WIDTH=250 -> 640x390
        self.screen_width = self.GRID_WIDTH + self.UI_WIDTH
        self.screen_height = self.GRID_HEIGHT
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400)) # Use fixed size surface
        self.clock = pygame.time.Clock()
        
        self.font_letter = pygame.font.Font(None, int(self.CELL_SIZE * 0.7))
        self.font_ui = pygame.font.Font(None, 24)
        self.font_ui_large = pygame.font.Font(None, 48)

        # Etc...
        self.np_random = None
        self.grid = None
        self.target_words = None
        self.word_metadata = None
        self.found_words_mask = None
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [0, 0]
        self.selection_start = None
        self.selection_end = None
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.flash_alpha = 0
        self.flash_color = (0, 0, 0)
        
        self.particles = []
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()


    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        # Select words
        self.target_words = self.np_random.choice(self.WORD_BANK, size=self.NUM_WORDS, replace=False).tolist()
        self.target_words.sort(key=len, reverse=True) # Place longer words first
        self.found_words_mask = [False] * self.NUM_WORDS
        self.word_metadata = {}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for i, word in enumerate(self.target_words):
            placed = False
            for _ in range(100): # 100 attempts to place a word
                self.np_random.shuffle(directions)
                direction = directions[0]
                
                start_x = self.np_random.integers(0, self.GRID_SIZE)
                start_y = self.np_random.integers(0, self.GRID_SIZE)
                
                end_x = start_x + (len(word) - 1) * direction[0]
                end_y = start_y + (len(word) - 1) * direction[1]

                if 0 <= end_x < self.GRID_SIZE and 0 <= end_y < self.GRID_SIZE:
                    can_place = True
                    for j in range(len(word)):
                        x = start_x + j * direction[0]
                        y = start_y + j * direction[1]
                        if self.grid[y][x] != '' and self.grid[y][x] != word[j]:
                            can_place = False
                            break
                    
                    if can_place:
                        for j in range(len(word)):
                            x = start_x + j * direction[0]
                            y = start_y + j * direction[1]
                            self.grid[y][x] = word[j]
                        
                        path = []
                        for j in range(len(word)):
                            path.append((start_x + j * direction[0], start_y + j * direction[1]))
                        self.word_metadata[word] = {'path': path, 'found': False}
                        
                        placed = True
                        break
            if not placed:
                pass # Word could not be placed, will be un-findable.

        # Fill empty cells
        alphabet = string.ascii_uppercase
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r][c] == '':
                    self.grid[r][c] = self.np_random.choice(list(alphabet))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selection_start = None
        self.selection_end = None
        
        self.last_space_held = False
        self.last_shift_held = False
        
        self.flash_alpha = 0
        self.flash_color = (0, 0, 0)
        
        self._generate_grid()
        
        self.particles = [
            [self.np_random.uniform(0, 640), self.np_random.uniform(0, 400), 
             self.np_random.uniform(0.5, 2), self.np_random.uniform(0.1, 0.3)] 
            for _ in range(50)
        ]

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _get_line_path(self, p1, p2):
        path = []
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1

        if not (x1 == x2 or y1 == y2 or abs(dx) == abs(dy)):
            return None

        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return [(x1, y1)]

        x_inc = dx / steps if steps != 0 else 0
        y_inc = dy / steps if steps != 0 else 0

        for i in range(steps + 1):
            path.append((round(x1 + i * x_inc), round(y1 + i * y_inc)))
        return path

    def _check_and_submit_word(self):
        reward = 0
        if self.selection_start is None or self.selection_end is None:
            return 0

        path = self._get_line_path(self.selection_start, self.selection_end)
        if path is None:
            self.flash_color = self.COLOR_FAIL_FLASH
            self.flash_alpha = 200
            reward -= 0.5
            return reward

        word_str = "".join([self.grid[y][x] for x, y in path])
        
        found = False
        for i, target in enumerate(self.target_words):
            if not self.found_words_mask[i] and (word_str == target or word_str[::-1] == target):
                self.found_words_mask[i] = True
                self.score += 10
                reward += 10
                self.word_metadata[target]['found'] = True
                # SFX: Word Found!
                self.flash_color = self.COLOR_SUCCESS_FLASH
                self.flash_alpha = 200
                found = True
                break
        
        if not found:
            reward -= 1.0
            # SFX: Word Invalid
            self.flash_color = self.COLOR_FAIL_FLASH
            self.flash_alpha = 200

        self.selection_start = None
        self.selection_end = None
        return reward

    def step(self, action):
        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = -0.01 # Small time penalty
        
        # Update game logic
        self.steps += 1
        self._update_particles()
        if self.flash_alpha > 0:
            self.flash_alpha = max(0, self.flash_alpha - 15)

        dx, dy = 0, 0
        if movement == 1: dy = -1
        elif movement == 2: dy = 1
        elif movement == 3: dx = -1
        elif movement == 4: dx = 1
        
        if dx != 0 or dy != 0:
            self.cursor_pos[0] = (self.cursor_pos[0] + dx + self.GRID_SIZE) % self.GRID_SIZE
            self.cursor_pos[1] = (self.cursor_pos[1] + dy + self.GRID_SIZE) % self.GRID_SIZE

        space_press = space_held and not self.last_space_held
        shift_press = shift_held and not self.last_shift_held

        if space_press:
            # SFX: Select
            if self.selection_start is None:
                self.selection_start = tuple(self.cursor_pos)
                self.selection_end = None
            else:
                self.selection_end = tuple(self.cursor_pos)
        
        if shift_press:
            if self.selection_start is not None and self.selection_end is not None:
                reward += self._check_and_submit_word()
            elif self.selection_start is not None:
                # SFX: Cancel
                self.selection_start = None
                self.selection_end = None

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        terminated = self._check_termination()
        if terminated and all(self.found_words_mask):
            reward += 50
            self.score += 50
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_termination(self):
        if self.steps >= self.MAX_STEPS:
            return True
        if all(self.found_words_mask):
            return True
        return False

    def _update_particles(self):
        for p in self.particles:
            p[1] -= p[3]
            if p[1] < 0:
                p[0] = self.np_random.uniform(0, 640)
                p[1] = 400

    def _render_game(self, surface):
        # Draw particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(surface, int(p[0]), int(p[1]), int(p[2]), (*self.COLOR_GRID_LINE, 50))

        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            pygame.draw.line(surface, self.COLOR_GRID_LINE, (i * self.CELL_SIZE, 0), (i * self.CELL_SIZE, self.GRID_HEIGHT))
            pygame.draw.line(surface, self.COLOR_GRID_LINE, (0, i * self.CELL_SIZE), (self.GRID_WIDTH, i * self.CELL_SIZE))

        # Draw found words background
        for word, meta in self.word_metadata.items():
            if meta.get('found', False):
                path = meta['path']
                for i in range(len(path) - 1):
                    p1, p2 = path[i], path[i+1]
                    px1, py1 = p1[0] * self.CELL_SIZE + self.CELL_SIZE // 2, p1[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                    px2, py2 = p2[0] * self.CELL_SIZE + self.CELL_SIZE // 2, p2[1] * self.CELL_SIZE + self.CELL_SIZE // 2
                    pygame.draw.line(surface, self.COLOR_FOUND_WORD_BG, (px1, py1), (px2, py2), self.CELL_SIZE // 2)
                for x, y in path:
                    pygame.gfxdraw.filled_circle(surface, x * self.CELL_SIZE + self.CELL_SIZE // 2, y * self.CELL_SIZE + self.CELL_SIZE // 2, self.CELL_SIZE // 3, self.COLOR_FOUND_WORD_BG)

        # Draw letters
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                letter = self.grid[r][c]
                is_found = any(meta.get('found', False) and (c,r) in meta['path'] for meta in self.word_metadata.values())
                color = self.COLOR_FOUND_WORD_LETTER if is_found else self.COLOR_LETTER
                text_surf = self.font_letter.render(letter, True, color)
                text_rect = text_surf.get_rect(center=(c * self.CELL_SIZE + self.CELL_SIZE // 2, r * self.CELL_SIZE + self.CELL_SIZE // 2))
                surface.blit(text_surf, text_rect)

        # Draw selection path
        if self.selection_start:
            end_pos = self.selection_end if self.selection_end else self.cursor_pos
            start_px = (self.selection_start[0] * self.CELL_SIZE + self.CELL_SIZE // 2, self.selection_start[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            end_px = (end_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2, end_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            pygame.draw.aaline(surface, self.COLOR_SELECTION_PATH, start_px, end_px, 2)
        
        if self.selection_start:
            cx, cy = self.selection_start
            center_px = (cx * self.CELL_SIZE + self.CELL_SIZE // 2, cy * self.CELL_SIZE + self.CELL_SIZE // 2)
            pygame.gfxdraw.aacircle(surface, center_px[0], center_px[1], self.CELL_SIZE // 2 - 2, self.COLOR_SELECTION_START)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(cx * self.CELL_SIZE, cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(surface, self.COLOR_CURSOR, cursor_rect, 2, border_radius=4)
        
    def _render_ui(self, surface):
        ui_rect = pygame.Rect(self.GRID_WIDTH, 0, 640 - self.GRID_WIDTH, 400)
        pygame.draw.rect(surface, self.COLOR_UI_BG, ui_rect)

        remaining_seconds = max(0, (self.MAX_STEPS - self.steps) // 30)
        timer_text = f"{remaining_seconds // 60:02d}:{remaining_seconds % 60:02d}"
        color = self.COLOR_UI_TIMER if remaining_seconds < 10 else self.COLOR_UI_TEXT
        text_surf = self.font_ui_large.render(timer_text, True, color)
        surface.blit(text_surf, (self.GRID_WIDTH + 20, 20))

        score_text = f"SCORE: {self.score}"
        text_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        surface.blit(text_surf, (self.GRID_WIDTH + 20, 80))

        y_offset = 130
        for i, word in enumerate(self.target_words):
            color = self.COLOR_UI_WORD_FOUND if self.found_words_mask[i] else self.COLOR_UI_TEXT
            text_surf = self.font_ui.render(word, True, color)
            surface.blit(text_surf, (self.GRID_WIDTH + 20, y_offset))
            if self.found_words_mask[i]:
                 pygame.draw.line(surface, self.COLOR_UI_WORD_FOUND, (self.GRID_WIDTH + 20, y_offset + 12), (self.GRID_WIDTH + 20 + text_surf.get_width(), y_offset + 12), 2)
            y_offset += 25
            if y_offset > 380: break

    def _render_effects(self, surface):
        if self.flash_alpha > 0:
            flash_surf = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
            flash_surf.fill((*self.flash_color, self.flash_alpha))
            surface.blit(flash_surf, (0, 0))

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Create sub-surfaces for scaling and layout to fit 640x400
        game_area_surf = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT))
        game_area_surf.fill(self.COLOR_BG)

        # Render all game elements
        self._render_game(game_area_surf)
        
        # Render UI overlay
        self._render_ui(self.screen) # Render UI directly on main screen
        
        # Blit game area onto main screen
        self.screen.blit(game_area_surf, (0, 0))

        # Render effects over everything
        self._render_effects(self.screen)
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
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
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    
    try:
        window = pygame.display.set_mode((640, 400))
        pygame.display.set_caption("Word Search")
    except pygame.error:
        window = None

    obs, info = env.reset()
    terminated = False
    
    running = True
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
        
        if window:
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            window.blit(surf, (0, 0))
            pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
            pygame.time.wait(2000)

        env.clock.tick(30)

    env.close()