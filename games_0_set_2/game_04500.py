
# Generated: 2025-08-28T02:36:20.572463
# Source Brief: brief_04500.md
# Brief Index: 4500

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a word's first letter, "
        "then press Shift at the word's last letter."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find hidden words in a grid of letters before time runs out. Correct words turn grey."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    GAME_DURATION_SECONDS = 60

    GRID_WIDTH = 18
    GRID_HEIGHT = 11
    WORD_COUNT = 15

    CELL_SIZE = 30
    GRID_ORIGIN_X = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_ORIGIN_Y = 75

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_LINES = (50, 60, 70)
    COLOR_LETTER = (220, 220, 230)
    COLOR_LETTER_FOUND = (100, 110, 120)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION = (70, 130, 180) # SteelBlue
    COLOR_UI_TEXT = (240, 240, 240)
    COLOR_SUCCESS = (0, 255, 127) # SpringGreen
    COLOR_FAIL = (255, 69, 0) # OrangeRed

    WORD_LIST = [
        "PYTHON", "GYMNASIUM", "AGENT", "REWARD", "ACTION", "STATE", "POLICY",
        "DEEP", "LEARNING", "SEARCH", "GRID", "FIND", "PUZZLE", "TIMER", "SOLVE",
        "CODE", "GAME", "VECTOR", "TENSOR", "KERAS", "VISUAL", "PIXEL", "FRAME"
    ]

    DIRECTIONS = {
        'E': (1, 0), 'W': (-1, 0), 'S': (0, 1), 'N': (0, -1),
        'SE': (1, 1), 'NW': (-1, -1), 'SW': (-1, 1), 'NE': (1, -1)
    }

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
        
        self.font_ui = pygame.font.Font(None, 36)
        self.font_grid = pygame.font.Font(None, 28)
        self.font_timer = pygame.font.Font(None, 48)

        # Game state variables are initialized in reset()
        self.grid = None
        self.solutions = None
        self.found_words = None
        self.cursor_pos = None
        self.first_letter_selection = None
        self.steps = None
        self.score = None
        self.time_remaining_frames = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.movement_cooldown = None
        self.feedback_effects = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining_frames = self.GAME_DURATION_SECONDS * self.FPS
        
        self._generate_grid()
        self.found_words = []
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.first_letter_selection = None
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.movement_cooldown = 0
        self.feedback_effects = []
        
        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.solutions = []
        
        num_to_place = min(self.WORD_COUNT, len(self.WORD_LIST))
        words_to_place = self.np_random.choice(self.WORD_LIST, size=num_to_place, replace=False).tolist()
        
        for word in words_to_place:
            placed = False
            for _ in range(200): # More attempts for robustness
                dir_key = self.np_random.choice(list(self.DIRECTIONS.keys()))
                dx, dy = self.DIRECTIONS[dir_key]
                
                start_x = self.np_random.integers(0, self.GRID_WIDTH)
                start_y = self.np_random.integers(0, self.GRID_HEIGHT)
                
                end_x = start_x + (len(word) - 1) * dx
                end_y = start_y + (len(word) - 1) * dy
                
                if not (0 <= end_x < self.GRID_WIDTH and 0 <= end_y < self.GRID_HEIGHT):
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
                        "end": (end_x, end_y),
                        "found": False
                    })
                    placed = True
                    break
        
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == '':
                    self.grid[y][x] = self.np_random.choice(list(alphabet))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        self.steps += 1
        self.time_remaining_frames -= 1
        if self.movement_cooldown > 0:
            self.movement_cooldown -= 1

        if self.movement_cooldown == 0:
            moved = True
            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            else: moved = False
            
            if moved:
                self.movement_cooldown = 3 # 3 frames cooldown
                self.cursor_pos[0] %= self.GRID_WIDTH
                self.cursor_pos[1] %= self.GRID_HEIGHT

        space_pressed = space_held and not self.prev_space_held
        if space_pressed:
            # sfx: select_letter
            self.first_letter_selection = tuple(self.cursor_pos)
            is_valid_endpoint = any(
                (self.first_letter_selection == sol['start'] or self.first_letter_selection == sol['end']) and not sol['found']
                for sol in self.solutions
            )
            reward += 0.1 if is_valid_endpoint else -0.01

        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.first_letter_selection:
            current_pos = tuple(self.cursor_pos)
            word_found = False
            for sol in self.solutions:
                if not sol['found']:
                    is_match = (self.first_letter_selection == sol['start'] and current_pos == sol['end']) or \
                               (self.first_letter_selection == sol['end'] and current_pos == sol['start'])
                    if is_match:
                        # sfx: word_found
                        sol['found'] = True
                        self.found_words.append(sol['word'])
                        self.score += 1
                        reward += 10
                        self._add_feedback_effect('line', sol['start'], sol['end'], self.COLOR_SUCCESS)
                        word_found = True
                        break
            
            if not word_found:
                # sfx: word_incorrect
                self._add_feedback_effect('flash', self.first_letter_selection, None, self.COLOR_FAIL)
            
            self.first_letter_selection = None
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        self._update_feedback_effects()
        
        terminated = False
        if self.score >= self.WORD_COUNT:
            # sfx: game_win
            reward += 50
            terminated = True
            self.game_over = True
        elif self.time_remaining_frames <= 0:
            # sfx: game_over
            reward = -100
            terminated = True
            self.game_over = True
            
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _add_feedback_effect(self, type, start_pos, end_pos, color):
        self.feedback_effects.append({
            "type": type,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "color": color,
            "timer": self.FPS, # 1 second duration
        })
        
    def _update_feedback_effects(self):
        self.feedback_effects = [eff for eff in self.feedback_effects if eff['timer'] > 0]
        for eff in self.feedback_effects:
            eff['timer'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(
                    self.GRID_ORIGIN_X + x * self.CELL_SIZE,
                    self.GRID_ORIGIN_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)

                is_found = any(sol['found'] and self._is_coord_in_word(x, y, sol) for sol in self.solutions)
                color = self.COLOR_LETTER_FOUND if is_found else self.COLOR_LETTER
                
                letter_surface = self.font_grid.render(self.grid[y][x], True, color)
                letter_rect = letter_surface.get_rect(center=rect.center)
                self.screen.blit(letter_surface, letter_rect)

        if self.first_letter_selection:
            x, y = self.first_letter_selection
            rect = pygame.Rect(
                self.GRID_ORIGIN_X + x * self.CELL_SIZE,
                self.GRID_ORIGIN_Y + y * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 3, border_radius=4)
        
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_ORIGIN_X + cx * self.CELL_SIZE,
            self.GRID_ORIGIN_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)

        for eff in self.feedback_effects:
            alpha = int(255 * (eff['timer'] / self.FPS))
            if eff['type'] == 'line':
                start_pixel = self._grid_to_pixel_center(eff['start_pos'])
                end_pixel = self._grid_to_pixel_center(eff['end_pos'])
                self._draw_line_alpha(self.screen, eff['color'] + (alpha,), start_pixel, end_pixel, 5)
            elif eff['type'] == 'flash':
                pos_pixel = self._grid_to_pixel_center(eff['start_pos'])
                radius = int(self.CELL_SIZE * 0.75 * (1 - (eff['timer'] / self.FPS)))
                self._draw_circle_alpha(self.screen, eff['color'] + (alpha,), pos_pixel, radius, 3)

    def _is_coord_in_word(self, x, y, solution):
        start_x, start_y = solution['start']
        end_x, end_y = solution['end']
        
        if not (min(start_x, end_x) <= x <= max(start_x, end_x) and \
                min(start_y, end_y) <= y <= max(start_y, end_y)):
            return False
        return (y - start_y) * (end_x - start_x) == (end_y - start_y) * (x - start_x)

    def _grid_to_pixel_center(self, grid_pos):
        x, y = grid_pos
        px = self.GRID_ORIGIN_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
        py = self.GRID_ORIGIN_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
        return (px, py)

    def _render_ui(self):
        score_text = f"Words: {self.score}/{self.WORD_COUNT}"
        score_surface = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surface, (20, 20))

        time_sec = math.ceil(max(0, self.time_remaining_frames) / self.FPS)
        time_text = f"{time_sec:02}"
        timer_color = self.COLOR_FAIL if time_sec <= 10 else self.COLOR_UI_TEXT
        time_surface = self.font_timer.render(time_text, True, timer_color)
        time_rect = time_surface.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(time_surface, time_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
        }

    def _draw_line_alpha(self, surface, color, start_pos, end_pos, width):
        x1, y1 = start_pos
        x2, y2 = end_pos
        lx, ly = abs(x1 - x2), abs(y1 - y2)
        temp_surf = pygame.Surface((lx + width*2, ly + width*2), pygame.SRCALPHA)
        local_start = (width if x1 > x2 else width, width if y1 > y2 else width)
        local_end = (lx + width if x1 < x2 else width, ly + width if y1 < y2 else width)
        pygame.draw.line(temp_surf, color, local_start, local_end, width)
        surface.blit(temp_surf, (min(x1, x2) - width, min(y1, y2) - width))

    def _draw_circle_alpha(self, surface, color, center, radius, width):
        if radius <= 0: return
        target_rect = pygame.Rect(center[0] - radius, center[1] - radius, radius * 2, radius * 2)
        temp_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
        for i in range(width):
            if radius - i > 0:
                pygame.gfxdraw.aacircle(temp_surf, radius, radius, radius - i, color)
        surface.blit(temp_surf, target_rect.topleft)

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    
    pygame.display.set_caption("Word Search")
    render_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    running = True
    terminated = False
    
    key_to_action = {
        pygame.K_UP: 1, pygame.K_w: 1,
        pygame.K_DOWN: 2, pygame.K_s: 2,
        pygame.K_LEFT: 3, pygame.K_a: 3,
        pygame.K_RIGHT: 4, pygame.K_d: 4,
    }
    
    while running:
        if terminated:
            print(f"Game Over! Final Info: {info}")
            obs, info = env.reset()
            terminated = False

        movement, space_held, shift_held = 0, 0, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        for key, move_action in key_to_action.items():
            if keys[key]:
                movement = move_action
                break
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}")
        
        render_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(render_surface, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(env.FPS)
        
    pygame.quit()