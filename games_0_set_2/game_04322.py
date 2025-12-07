
# Generated: 2025-08-28T02:03:23.915365
# Source Brief: brief_04322.md
# Brief Index: 4322

        
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
        "Controls: Arrow keys to move the cursor. "
        "Press Space to mark the start of a word. "
        "Press Shift to mark the end and submit the word."
    )

    game_description = (
        "Find all the hidden words in the grid before the time runs out. "
        "Levels get harder with bigger grids and longer words. "
        "Incorrect guesses will cost you time!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_AREA_WIDTH = 400
        self.UI_AREA_WIDTH = self.SCREEN_WIDTH - self.GRID_AREA_WIDTH
        self.FPS = 30

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font_grid = pygame.font.SysFont("Consolas", 16, bold=True)
        self.font_ui = pygame.font.SysFont("Arial", 18)
        self.font_ui_bold = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_gameover = pygame.font.SysFont("Arial", 48, bold=True)

        self._init_colors()
        self._init_word_bank()

        self.np_random = None
        self.grid = []
        self.grid_size = 0
        self.words_to_find = []
        self.found_words = []
        self.word_locations = {}
        self.found_word_paths = []
        self.cursor_pos = [0, 0]
        self.selection_start = None
        self.selection_active = False
        self.particles = []
        self.level = 1
        self.score = 0
        self.steps = 0
        self.timer = 0.0
        self.game_over = False
        self.game_over_message = ""
        self.prev_action = np.array([0, 0, 0])
        
        self.reset()
        self.validate_implementation()

    def _init_colors(self):
        self.COLOR_BG = (15, 25, 35)
        self.COLOR_GRID_BG = (25, 40, 55)
        self.COLOR_UI_BG = (20, 30, 45)
        self.COLOR_LETTER = (200, 220, 255)
        self.COLOR_CURSOR = (255, 200, 0, 100)
        self.COLOR_SELECTION_START = (0, 150, 255)
        self.COLOR_SELECTION_LINE = (0, 200, 255)
        self.COLOR_FOUND_WORD_BG = (0, 80, 40, 150)
        self.COLOR_FOUND_WORD_STRIKE = (50, 255, 150)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_HEADER = (255, 200, 0)
        self.COLOR_TIMER_NORMAL = (50, 255, 150)
        self.COLOR_TIMER_LOW = (255, 80, 80)
        self.COLOR_INCORRECT_FLASH = (255, 0, 0, 150)

    def _init_word_bank(self):
        self.WORD_BANK = {
            4: ["CODE", "GAME", "GRID", "GYMS", "LOOP", "PLAY", "STEP", "TEST", "TIME", "WORD"],
            5: ["AGENT", "ARRAY", "BOXES", "CLOCK", "EVENT", "FRAME", "LOGIC", "PIXEL", "PROXY", "RESET"],
            6: ["ACTION", "CHOICE", "CURSOR", "RENDER", "REWARD", "SCREEN", "SPACES", "SPRITE", "TENSOR", "VISUAL"],
            7: ["CONTROL", "EPISODE", "EXPLORE", "LETTERS", "METADATA", "PYGAME", "RANDOM", "SUCCESS", "TIMEOUT", "VECTOR"],
            8: ["BOUNDARY", "CALLBACK", "DISCRETE", "FEEDBACK", "GRAPHICS", "MONITOR", "OBSERVE", "PENALTY", "TERMINAL", "TRAINING"],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            random.seed(seed)

        self.level = 1
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.game_over_message = ""
        self.prev_action = np.array([0, 0, 0])
        self.particles = []

        self._setup_level()
        return self._get_observation(), self._get_info()

    def _setup_level(self):
        self.timer = 60.0
        self.words_to_find = []
        self.found_words = []
        self.word_locations = {}
        self.found_word_paths = []
        self.selection_start = None
        self.selection_active = False
        self.incorrect_flash_timer = 0

        self.grid_size = min(20, 10 + (self.level - 1) * 2)
        word_len = min(8, 4 + (self.level - 1))
        
        self._generate_word_list(word_len)
        self._generate_grid()
        self.cursor_pos = [self.grid_size // 2, self.grid_size // 2]

    def _generate_word_list(self, length):
        available_words = self.WORD_BANK.get(length, self.WORD_BANK[max(self.WORD_BANK.keys())])
        self.words_to_find = random.sample(available_words, min(len(available_words), 15))

    def _generate_grid(self):
        for _ in range(10): # Max 10 attempts to generate a valid grid
            self.grid = [['' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            self.word_locations.clear()
            
            words_to_place = list(self.words_to_find)
            random.shuffle(words_to_place)
            
            possible = True
            for word in words_to_place:
                if random.random() < 0.5: word = word[::-1]
                if not self._try_place_word(word):
                    possible = False
                    break
            
            if possible:
                # Fill rest of grid with random letters
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        if self.grid[r][c] == '':
                            self.grid[r][c] = random.choice(string.ascii_uppercase)
                return
        
        # If we failed 10 times, create a dummy grid (should be rare)
        print(f"Warning: Failed to generate grid for level {self.level}. Creating dummy grid.")
        self.words_to_find = [] # No words to find, level auto-completes

    def _try_place_word(self, word):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        random.shuffle(directions)
        
        for _ in range(100): # 100 placement attempts for this word
            r, c = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
            dr, dc = random.choice(directions)
            
            end_r, end_c = r + (len(word) - 1) * dr, c + (len(word) - 1) * dc
            
            if not (0 <= end_r < self.grid_size and 0 <= end_c < self.grid_size):
                continue
            
            can_place = True
            for i in range(len(word)):
                curr_r, curr_c = r + i * dr, c + i * dc
                if self.grid[curr_r][curr_c] not in ('', word[i]):
                    can_place = False
                    break
            
            if can_place:
                for i in range(len(word)):
                    curr_r, curr_c = r + i * dr, c + i * dc
                    self.grid[curr_r][curr_c] = word[i]
                self.word_locations[word.replace(' ', '')] = ((c, r), (end_c, end_r))
                self.word_locations[word.replace(' ', '')[::-1]] = ((end_c, end_r), (c, r))
                return True
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.timer = max(0, self.timer - 1.0 / self.FPS)
        if self.incorrect_flash_timer > 0:
            self.incorrect_flash_timer -= 1

        reward += self._handle_input(action)

        terminated = False
        if not self.words_to_find:
            reward += 100
            self.level += 1
            self._setup_level()
            # In a continuous play scenario, we'd just continue.
            # For typical RL, completing a level is a terminal success state.
            self.game_over = True
            terminated = True
            self.game_over_message = "LEVEL CLEAR!"
        elif self.timer <= 0:
            reward -= 100
            self.game_over = True
            terminated = True
            self.game_over_message = "TIME'S UP!"

        self.prev_action = action
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_val, shift_val = action
        prev_space_val = self.prev_action[1]
        prev_shift_val = self.prev_action[2]

        reward = 0
        dist_before = self._get_min_dist_to_word()

        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.grid_size - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.grid_size - 1)

        dist_after = self._get_min_dist_to_word()
        if dist_after < dist_before:
            reward += 0.1

        space_pressed = space_val == 1 and prev_space_val == 0
        shift_pressed = shift_val == 1 and prev_shift_val == 0

        if space_pressed:
            # Sfx: select_start.wav
            self.selection_start = list(self.cursor_pos)
            self.selection_active = True

        if shift_pressed and self.selection_active:
            # Sfx: submit.wav
            reward += self._check_word(self.selection_start, self.cursor_pos)
            self.selection_active = False
            self.selection_start = None

        return reward

    def _check_word(self, start_pos, end_pos):
        start_c, start_r = start_pos
        end_c, end_r = end_pos
        
        dc, dr = end_c - start_c, end_r - start_r
        
        if not (dc == 0 or dr == 0 or abs(dc) == abs(dr)):
            self.timer = max(0, self.timer - 2.0)
            self.incorrect_flash_timer = 10 # frames
            return 0 # Not a straight line

        step_c, step_r = np.sign(dc), np.sign(dr)
        length = max(abs(dc), abs(dr)) + 1
        
        word = ""
        path = []
        for i in range(length):
            c = start_c + i * step_c
            r = start_r + i * step_r
            word += self.grid[r][c]
            path.append((c, r))

        if word in self.words_to_find:
            # Sfx: correct_word.wav
            self.words_to_find.remove(word)
            self.found_words.append(word)
            self.score += 10
            self.found_word_paths.append(path)
            self._create_particles_for_word(path)
            return 10
        else:
            # Sfx: incorrect_word.wav
            self.timer = max(0, self.timer - 2.0)
            self.incorrect_flash_timer = 10 # frames
            return 0

    def _get_min_dist_to_word(self):
        if not self.words_to_find:
            return 0
        min_dist = float('inf')
        cursor_c, cursor_r = self.cursor_pos
        for word in self.words_to_find:
            if word in self.word_locations:
                start_pos, _ = self.word_locations[word]
                dist = abs(cursor_c - start_pos[0]) + abs(cursor_r - start_pos[1])
                min_dist = min(min_dist, dist)
        return min_dist

    def _create_particles_for_word(self, path):
        # Find midpoint of the word
        mid_idx = len(path) // 2
        center_c, center_r = path[mid_idx]
        cell_size = self.GRID_AREA_WIDTH / self.grid_size
        px = (center_c + 0.5) * cell_size
        py = (center_r + 0.5) * cell_size

        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = random.randint(15, 30)
            color = random.choice([(255, 255, 100), (255, 200, 50), (255, 255, 255)])
            self.particles.append({'pos': [px, py], 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_surface = self.screen.subsurface(pygame.Rect(0, 0, self.GRID_AREA_WIDTH, self.SCREEN_HEIGHT))
        grid_surface.fill(self.COLOR_GRID_BG)
        
        cell_size = self.GRID_AREA_WIDTH / self.grid_size
        
        # Draw found word backgrounds
        for path in self.found_word_paths:
            for c, r in path:
                bg_rect = pygame.Rect(c * cell_size, r * cell_size, cell_size, cell_size)
                s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                s.fill(self.COLOR_FOUND_WORD_BG)
                grid_surface.blit(s, bg_rect.topleft)

        # Draw letters
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                letter = self.grid[r][c]
                text_surf = self.font_grid.render(letter, True, self.COLOR_LETTER)
                text_rect = text_surf.get_rect(center=(c * cell_size + cell_size / 2, r * cell_size + cell_size / 2))
                grid_surface.blit(text_surf, text_rect)
        
        # Draw found word strikethroughs
        for path in self.found_word_paths:
            start_c, start_r = path[0]
            end_c, end_r = path[-1]
            start_px = (start_c + 0.5) * cell_size
            start_py = (start_r + 0.5) * cell_size
            end_px = (end_c + 0.5) * cell_size
            end_py = (end_r + 0.5) * cell_size
            pygame.draw.line(grid_surface, self.COLOR_FOUND_WORD_STRIKE, (start_px, start_py), (end_px, end_py), 3)

        # Draw cursor
        cursor_c, cursor_r = self.cursor_pos
        cursor_rect = pygame.Rect(cursor_c * cell_size, cursor_r * cell_size, cell_size, cell_size)
        s = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        grid_surface.blit(s, cursor_rect.topleft)

        # Draw selection
        if self.selection_active and self.selection_start:
            start_c, start_r = self.selection_start
            start_px = start_c * cell_size + cell_size / 2
            start_py = start_r * cell_size + cell_size / 2
            pygame.gfxdraw.filled_circle(grid_surface, int(start_px), int(start_py), int(cell_size/4), self.COLOR_SELECTION_START)
            
            cursor_px = cursor_c * cell_size + cell_size / 2
            cursor_py = cursor_r * cell_size + cell_size / 2
            pygame.draw.line(grid_surface, self.COLOR_SELECTION_LINE, (start_px, start_py), (cursor_px, cursor_py), 3)

        # Draw incorrect flash
        if self.incorrect_flash_timer > 0:
            s = pygame.Surface((self.GRID_AREA_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(self.incorrect_flash_timer / 10 * self.COLOR_INCORRECT_FLASH[3])
            s.fill((self.COLOR_INCORRECT_FLASH[0], self.COLOR_INCORRECT_FLASH[1], self.COLOR_INCORRECT_FLASH[2], alpha))
            grid_surface.blit(s, (0, 0))

        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifespan'] / 30))
                pygame.gfxdraw.filled_circle(grid_surface, int(p['pos'][0]), int(p['pos'][1]), 2, p['color'] + (alpha,))

    def _render_ui(self):
        ui_surface = self.screen.subsurface(pygame.Rect(self.GRID_AREA_WIDTH, 0, self.UI_AREA_WIDTH, self.SCREEN_HEIGHT))
        ui_surface.fill(self.COLOR_UI_BG)
        
        y_offset = 15
        
        # Level and Score
        level_text = self.font_title.render(f"LEVEL {self.level}", True, self.COLOR_UI_HEADER)
        ui_surface.blit(level_text, (15, y_offset))
        y_offset += 35
        
        score_text = self.font_ui_bold.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        ui_surface.blit(score_text, (15, y_offset))
        y_offset += 25
        
        # Timer
        timer_color = self.COLOR_TIMER_NORMAL if self.timer > 10 else self.COLOR_TIMER_LOW
        timer_text = self.font_ui_bold.render(f"TIME: {self.timer:.1f}", True, timer_color)
        ui_surface.blit(timer_text, (15, y_offset))
        y_offset += 40
        
        # Words to find
        words_header = self.font_ui_bold.render("WORDS TO FIND:", True, self.COLOR_UI_HEADER)
        ui_surface.blit(words_header, (15, y_offset))
        y_offset += 25
        
        for word in self.words_to_find:
            word_surf = self.font_ui.render(word, True, self.COLOR_UI_TEXT)
            ui_surface.blit(word_surf, (25, y_offset))
            y_offset += 20
        
        for word in self.found_words:
            word_surf = self.font_ui.render(word, True, (100, 120, 140))
            ui_surface.blit(word_surf, (25, y_offset))
            # Draw strikethrough
            line_y = y_offset + word_surf.get_height() // 2
            pygame.draw.line(ui_surface, (100, 120, 140), (25, line_y), (25 + word_surf.get_width(), line_y), 1)
            y_offset += 20

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        text_surf = self.font_gameover.render(self.game_over_message, True, self.COLOR_UI_HEADER)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "words_remaining": len(self.words_to_find),
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
        
        # Test survival
        self.reset()
        for _ in range(50):
            _, _, term, _, _ = self.step(self.action_space.sample())
            if term: break
        assert self.steps >= 50 or term, "Agent should survive at least 50 steps from random actions"

        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to run the file directly to play the game
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    game_over_display_timer = 0
    
    # Store the state of keys
    keys = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_SPACE: False,
        pygame.K_LSHIFT: False,
        pygame.K_RSHIFT: False,
    }

    # Use a Pygame screen for human rendering
    pygame.display.set_caption("Word Search Environment")
    human_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

    while running:
        # Construct action from keyboard state
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = np.array([movement, space_held, shift_held])

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render to the human-visible screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        human_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Level: {info['level']}")
            game_over_display_timer = env.FPS * 3 # Display for 3 seconds

        if game_over_display_timer > 0:
            game_over_display_timer -= 1
            if game_over_display_timer == 0:
                print("Resetting game...")
                env.reset()

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys:
                    keys[event.key] = True
                if event.key == pygame.K_r: # Manual reset
                    print("Manual reset.")
                    env.reset()
            if event.type == pygame.KEYUP:
                if event.key in keys:
                    keys[event.key] = False

        env.clock.tick(env.FPS)
        
    env.close()