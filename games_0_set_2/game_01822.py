
# Generated: 2025-08-27T18:24:26.567222
# Source Brief: brief_01822.md
# Brief Index: 1822

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select the first letter of a word, "
        "and Shift to select the last letter to submit it."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find hidden words in a grid of letters before time runs out. "
        "Correct words increase your score, but wrong guesses cost you time."
    )

    # Frames auto-advance for the real-time timer.
    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = (14, 10) # Columns, Rows
    CELL_SIZE = 32
    GRID_TOP_LEFT = (40, 40)
    
    # Colors
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_LINE = (50, 65, 80)
    COLOR_LETTER = (220, 220, 230)
    COLOR_FOUND_LETTER = (80, 90, 100)
    COLOR_CURSOR = (255, 180, 0)
    COLOR_SELECTION = (0, 150, 255)
    COLOR_CORRECT = (0, 255, 120)
    COLOR_INCORRECT = (255, 80, 80)
    COLOR_UI_TEXT = (200, 200, 210)
    COLOR_UI_VALUE = (255, 255, 255)
    
    # Word list
    MASTER_WORD_LIST = [
        "PYTHON", "AGENT", "REWARD", "STATE", "ACTION", "POLICY", "GRID", "WORLD",
        "LEARN", "MODEL", "SOLVE", "SEARCH", "PLAY", "GAME", "STEP", "RESET", "DEEP",
        "CODE", "SPACE", "TIMER", "SCORE", "FIND", "WORD"
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
        
        self.font_letter = pygame.font.Font(None, 28)
        self.font_ui_header = pygame.font.Font(None, 20)
        self.font_ui_value = pygame.font.Font(None, 26)
        self.font_word_list = pygame.font.Font(None, 22)

        self.grid = []
        self.words_to_find = []
        self.word_locations = {}
        self.cursor_pos = [0, 0]
        self.first_letter_selection = None
        self.found_words = set()
        self.feedback_effects = []
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.max_steps = 120 * 30  # 120 seconds at 30fps
        self.timer = self.max_steps

        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.max_steps
        self.cursor_pos = [self.GRID_SIZE[0] // 2, self.GRID_SIZE[1] // 2]
        self.first_letter_selection = None
        self.found_words = set()
        self.feedback_effects = []
        self.prev_space_held = False
        self.prev_shift_held = False

        self._generate_grid_and_words()
        
        return self._get_observation(), self._get_info()

    def _generate_grid_and_words(self):
        self.grid = [['' for _ in range(self.GRID_SIZE[0])] for _ in range(self.GRID_SIZE[1])]
        self.words_to_find = sorted(list(self.np_random.choice(self.MASTER_WORD_LIST, size=5, replace=False)))
        self.word_locations = {}

        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]

        for word in self.words_to_find:
            placed = False
            for _ in range(100): # 100 attempts to place a word
                word_to_place = word if self.np_random.random() < 0.5 else word[::-1]
                direction = self.np_random.choice(len(directions))
                dx, dy = directions[direction]
                
                x_start = self.np_random.integers(0, self.GRID_SIZE[0])
                y_start = self.np_random.integers(0, self.GRID_SIZE[1])
                
                x_end = x_start + (len(word_to_place) - 1) * dx
                y_end = y_start + (len(word_to_place) - 1) * dy

                if not (0 <= x_end < self.GRID_SIZE[0] and 0 <= y_end < self.GRID_SIZE[1]):
                    continue

                can_place = True
                for i in range(len(word_to_place)):
                    x, y = x_start + i * dx, y_start + i * dy
                    if self.grid[y][x] != '' and self.grid[y][x] != word_to_place[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i in range(len(word_to_place)):
                        x, y = x_start + i * dx, y_start + i * dy
                        self.grid[y][x] = word_to_place[i]
                    
                    # Store original word with its start/end points
                    self.word_locations[word] = ( (x_start, y_start), (x_end, y_end) )
                    placed = True
                    break
            
            if not placed:
                # This should be rare with a large enough grid
                pass 

        # Fill empty cells with random letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                if self.grid[y][x] == '':
                    self.grid[y][x] = self.np_random.choice(list(alphabet))

    def step(self, action):
        movement = action[0]
        space_pressed = action[1] == 1 and not self.prev_space_held
        shift_pressed = action[2] == 1 and not self.prev_shift_held
        self.prev_space_held = action[1] == 1
        self.prev_shift_held = action[2] == 1
        
        reward = 0
        self.game_over = False

        # --- Handle cursor movement ---
        prev_cursor_pos = list(self.cursor_pos)
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE[0] - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE[1] - 1)

        # Continuous reward for moving towards a goal
        if tuple(prev_cursor_pos) != tuple(self.cursor_pos):
            reward += self._calculate_movement_reward(prev_cursor_pos)

        # --- Handle selections ---
        if space_pressed:
            self.first_letter_selection = tuple(self.cursor_pos)
            # SFX: select_start.wav

        if shift_pressed and self.first_letter_selection:
            second_letter_selection = tuple(self.cursor_pos)
            
            found_match = False
            for word, (start, end) in self.word_locations.items():
                if word in self.found_words:
                    continue
                
                if (self.first_letter_selection == start and second_letter_selection == end) or \
                   (self.first_letter_selection == end and second_letter_selection == start):
                    self.found_words.add(word)
                    self.score += 10
                    reward += 10
                    self._create_feedback_effect(word, self.COLOR_CORRECT)
                    found_match = True
                    # SFX: correct_word.wav
                    break
            
            if not found_match:
                reward -= 1
                self.timer = max(0, self.timer - 10 * 30) # 10 second penalty
                self._create_feedback_effect_at_cursor(self.COLOR_INCORRECT)
                # SFX: incorrect_selection.wav

            self.first_letter_selection = None

        # --- Update game state ---
        self.steps += 1
        self.timer -= 1
        self._update_effects()

        # --- Check for termination ---
        if self.timer <= 0:
            self.game_over = True
            reward -= 10 # Penalty for running out of time
        
        if len(self.found_words) == len(self.words_to_find):
            self.game_over = True
            reward += 50 # Bonus for completing the puzzle
            self.score += 50
        
        terminated = self.game_over or self.steps >= self.max_steps
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_movement_reward(self, prev_pos):
        unfound_starts = [loc[0] for word, loc in self.word_locations.items() if word not in self.found_words]
        if not unfound_starts:
            return 0
        
        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        closest_target = min(unfound_starts, key=lambda p: dist(prev_pos, p))
        
        dist_before = dist(prev_pos, closest_target)
        dist_after = dist(self.cursor_pos, closest_target)
        
        return (dist_before - dist_after) * 0.1

    def _create_feedback_effect(self, word, color):
        (x_start, y_start), (x_end, y_end) = self.word_locations[word]
        num_steps = max(abs(x_end - x_start), abs(y_end - y_start))
        
        for i in range(num_steps + 1):
            t = i / num_steps if num_steps > 0 else 0
            x = int(x_start + t * (x_end - x_start))
            y = int(y_start + t * (y_end - y_start))
            
            for _ in range(5): # 5 particles per letter
                px, py = self._grid_to_pixel(x, y)
                angle = self.np_random.random() * 2 * math.pi
                speed = 2 + self.np_random.random() * 2
                self.feedback_effects.append({
                    "pos": [px, py],
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "life": 20 + self.np_random.integers(0, 10),
                    "color": color,
                    "radius": 3 + self.np_random.random()
                })

    def _create_feedback_effect_at_cursor(self, color):
        px, py = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        for _ in range(20):
            angle = self.np_random.random() * 2 * math.pi
            speed = 1 + self.np_random.random() * 3
            self.feedback_effects.append({
                "pos": [px, py],
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "life": 25 + self.np_random.integers(0, 15),
                "color": color,
                "radius": 2 + self.np_random.random() * 2
            })
    
    def _update_effects(self):
        for p in self.feedback_effects:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][1] += 0.1 # Gravity
            p["life"] -= 1
        self.feedback_effects = [p for p in self.feedback_effects if p["life"] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _grid_to_pixel(self, x, y):
        return (
            self.GRID_TOP_LEFT[0] + x * self.CELL_SIZE + self.CELL_SIZE // 2,
            self.GRID_TOP_LEFT[1] + y * self.CELL_SIZE + self.CELL_SIZE // 2
        )

    def _render_game(self):
        # Determine which cells are part of found words
        found_cells = set()
        for word in self.found_words:
            (x_start, y_start), (x_end, y_end) = self.word_locations[word]
            dx = np.sign(x_end - x_start)
            dy = np.sign(y_end - y_start)
            steps = max(abs(x_end-x_start), abs(y_end-y_start))
            for i in range(steps + 1):
                found_cells.add((x_start + i * dx, y_start + i * dy))

        # Render grid letters
        for y in range(self.GRID_SIZE[1]):
            for x in range(self.GRID_SIZE[0]):
                px, py = self._grid_to_pixel(x, y)
                color = self.COLOR_FOUND_LETTER if (x, y) in found_cells else self.COLOR_LETTER
                letter_surf = self.font_letter.render(self.grid[y][x], True, color)
                self.screen.blit(letter_surf, letter_surf.get_rect(center=(px, py)))

        # Render cursor and selection line
        cursor_px, cursor_py = self._grid_to_pixel(self.cursor_pos[0], self.cursor_pos[1])
        pulse = (math.sin(self.steps * 0.2) + 1) / 2 * 4
        pygame.gfxdraw.rectangle(self.screen, 
                                 (cursor_px - self.CELL_SIZE//2, cursor_py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE),
                                 (*self.COLOR_CURSOR, 100))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, 
                         (cursor_px - self.CELL_SIZE//2, cursor_py - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE), 
                         width=int(2 + pulse/2))

        if self.first_letter_selection:
            start_px, start_py = self._grid_to_pixel(self.first_letter_selection[0], self.first_letter_selection[1])
            pygame.draw.aaline(self.screen, self.COLOR_SELECTION, (start_px, start_py), (cursor_px, cursor_py), 2)
            pygame.gfxdraw.filled_circle(self.screen, start_px, start_py, 6, self.COLOR_SELECTION)
            pygame.gfxdraw.aacircle(self.screen, start_px, start_py, 6, self.COLOR_SELECTION)
        
        # Render feedback effects
        for p in self.feedback_effects:
            alpha = int(255 * (p['life'] / 40))
            color = (*p['color'], alpha)
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), color)

    def _render_ui(self):
        # UI Panel for words
        word_panel_x = self.GRID_TOP_LEFT[0] + self.GRID_SIZE[0] * self.CELL_SIZE + 20
        word_panel_y = self.GRID_TOP_LEFT[1]
        
        header_surf = self.font_ui_header.render("WORDS TO FIND", True, self.COLOR_UI_TEXT)
        self.screen.blit(header_surf, (word_panel_x, word_panel_y))

        for i, word in enumerate(self.words_to_find):
            y_pos = word_panel_y + 30 + i * 25
            is_found = word in self.found_words
            color = self.COLOR_FOUND_LETTER if is_found else self.COLOR_UI_VALUE
            
            self.font_word_list.set_strikethrough(is_found)
            word_surf = self.font_word_list.render(word, True, color)
            self.screen.blit(word_surf, (word_panel_x, y_pos))
        self.font_word_list.set_strikethrough(False) # Reset strikethrough

        # Top bar for Score and Time
        def draw_top_bar_item(label, value, x_pos, width):
            pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, (x_pos, 5, width, 28), border_radius=5)
            label_surf = self.font_ui_header.render(label, True, self.COLOR_UI_TEXT)
            self.screen.blit(label_surf, (x_pos + 10, 12))
            value_surf = self.font_ui_value.render(value, True, self.COLOR_UI_VALUE)
            self.screen.blit(value_surf, value_surf.get_rect(right=x_pos + width - 10, centery=19))

        # Score
        draw_top_bar_item("SCORE", f"{self.score}", 10, 150)
        
        # Time
        time_left_sec = max(0, self.timer / 30)
        time_color = self.COLOR_INCORRECT if time_left_sec < 10 else self.COLOR_UI_VALUE
        
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINE, (170, 5, self.SCREEN_WIDTH - 180, 28), border_radius=5)
        time_label_surf = self.font_ui_header.render("TIME", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_label_surf, (180, 12))
        
        time_bar_width = (self.SCREEN_WIDTH - 180 - 60)
        time_ratio = max(0, self.timer / self.max_steps)
        current_time_width = int(time_bar_width * time_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_BG, (230, 10, time_bar_width, 18), border_radius=3)
        pygame.draw.rect(self.screen, time_color, (230, 10, current_time_width, 18), border_radius=3)


    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": max(0, self.timer / 30),
            "words_found": len(self.found_words),
            "words_total": len(self.words_to_find)
        }

    def close(self):
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

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Grid Search")
    clock = pygame.time.Clock()
    running = True

    action = env.action_space.sample()
    action.fill(0)

    print(env.user_guide)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Human controls ---
        keys = pygame.key.get_pressed()
        
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render the observation to the display screen ---
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            obs, info = env.reset()
        
        clock.tick(30) # Run at 30 FPS

    env.close()