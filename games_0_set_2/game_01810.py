
# Generated: 2025-08-27T18:22:45.323759
# Source Brief: brief_01810.md
# Brief Index: 1810

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to mark the start of a word. "
        "Move the cursor to the end of the word and press Space again to submit. "
        "Press Shift to cancel a selection."
    )

    game_description = (
        "Find hidden words in the grid before time runs out. Select words by marking their start and end "
        "letters. Correctly identified words will be highlighted and removed from the list."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 60
    GAME_DURATION_SECONDS = 90
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    GRID_SIZE = 15
    GRID_AREA_WIDTH = 400
    CELL_SIZE = GRID_AREA_WIDTH // GRID_SIZE
    GRID_OFFSET_X = (GRID_AREA_WIDTH - (GRID_SIZE * CELL_SIZE)) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - (GRID_SIZE * CELL_SIZE)) // 2

    # --- Colors ---
    COLOR_BG = (25, 28, 32)
    COLOR_GRID_BG = (35, 38, 43)
    COLOR_GRID_LINES = (50, 55, 61)
    COLOR_TEXT_NORMAL = (210, 215, 220)
    COLOR_TEXT_DIM = (100, 105, 110)
    COLOR_TEXT_SUCCESS = (100, 220, 120)
    COLOR_CURSOR = (255, 190, 80)
    COLOR_SELECTION = (255, 190, 80, 100)
    COLOR_FOUND_WORD_BG = (80, 160, 90, 120)
    COLOR_INCORRECT_FLASH = (220, 80, 80, 180)
    
    PARTICLE_COLORS = [(255, 190, 80), (255, 220, 120), (255, 255, 255)]

    WORD_LIST = [
        "GRID", "WORD", "FIND", "GAME", "TIME", "PLAY", "CODE", "PYTHON",
        "AGENT", "STEP", "RESET", "ACTION", "STATE", "REWARD", "SCORE"
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
        
        self.font_grid = pygame.font.SysFont("Consolas", int(self.CELL_SIZE * 0.6), bold=True)
        self.font_ui_title = pygame.font.SysFont("Segoe UI", 20, bold=True)
        self.font_ui_body = pygame.font.SysFont("Segoe UI", 16)
        self.font_ui_timer = pygame.font.SysFont("Consolas", 32, bold=True)
        self.font_game_over = pygame.font.SysFont("Segoe UI", 50, bold=True)

        self.grid = np.array([['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)])
        self.solution_grid = np.array([[-1 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)])
        self.solutions = {}
        self.found_words_info = {}
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selection_start_pos = None
        self.time_remaining = 0.0
        self.found_words = set()
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.incorrect_flash_timer = 0
        
        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def _generate_grid(self):
        self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), fill_value=' ', dtype='<U1')
        self.solution_grid = np.full((self.GRID_SIZE, self.GRID_SIZE), fill_value=-1, dtype=int)
        self.solutions = {}
        
        directions = [(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)]
        
        for i, word in enumerate(self.WORD_LIST):
            placed = False
            for _ in range(200): # Attempts to place a word
                random.shuffle(directions)
                d = random.choice(directions)
                dr, dc = d
                
                start_r = self.np_random.integers(0, self.GRID_SIZE)
                start_c = self.np_random.integers(0, self.GRID_SIZE)
                
                end_r = start_r + (len(word) - 1) * dr
                end_c = start_c + (len(word) - 1) * dc
                
                if not (0 <= end_r < self.GRID_SIZE and 0 <= end_c < self.GRID_SIZE):
                    continue

                can_place = True
                for j in range(len(word)):
                    r, c = start_r + j * dr, start_c + j * dc
                    if self.grid[r, c] != ' ' and self.grid[r, c] != word[j]:
                        can_place = False
                        break
                
                if can_place:
                    for j in range(len(word)):
                        r, c = start_r + j * dr, start_c + j * dc
                        self.grid[r, c] = word[j]
                        self.solution_grid[r, c] = i
                    self.solutions[word] = ((start_r, start_c), (end_r, end_c))
                    placed = True
                    break
        
        # Fill remaining spaces
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == ' ':
                    self.grid[r, c] = chr(self.np_random.integers(65, 91)) # A-Z

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_grid()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selection_start_pos = None
        self.time_remaining = self.GAME_DURATION_SECONDS
        self.found_words = set()
        self.found_words_info = {}
        self.prev_space_held = False
        self.prev_shift_held = False
        self.particles = []
        self.incorrect_flash_timer = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        self.game_over = self.time_remaining <= 0 or len(self.found_words) == len(self.WORD_LIST)
        if self.game_over:
            if len(self.found_words) == len(self.WORD_LIST):
                reward += 100 # Final bonus for winning
            return self._get_observation(), reward, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self.steps += 1
        self.time_remaining -= 1.0 / self.FPS
        
        # --- Handle Input ---
        # Movement
        if movement == 1: self.cursor_pos[0] -= 1  # Up
        elif movement == 2: self.cursor_pos[0] += 1  # Down
        elif movement == 3: self.cursor_pos[1] -= 1  # Left
        elif movement == 4: self.cursor_pos[1] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # Shift (Cancel Selection)
        if shift_held and not self.prev_shift_held:
            if self.selection_start_pos is not None:
                self.selection_start_pos = None
                # SFX: Cancel sound

        # Space (Select/Submit)
        if space_held and not self.prev_space_held:
            if self.selection_start_pos is None:
                self.selection_start_pos = tuple(self.cursor_pos)
                # SFX: Select start sound
            else:
                reward += self._submit_word()
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Continuous reward for selection
        if self.selection_start_pos:
            line_cells = self._get_line_cells(self.selection_start_pos, self.cursor_pos)
            for r, c in line_cells:
                word_idx = self.solution_grid[r, c]
                if word_idx != -1 and self.WORD_LIST[word_idx] not in self.found_words:
                    reward += 0.1
                else:
                    reward -= 0.1

        self.score += reward
        self._update_particles()
        if self.incorrect_flash_timer > 0:
            self.incorrect_flash_timer -= 1
        
        terminated = self.time_remaining <= 0 or len(self.found_words) == len(self.WORD_LIST) or self.steps >= self.MAX_STEPS
        if terminated and len(self.found_words) == len(self.WORD_LIST) and not self.game_over:
             reward += 100 # Win bonus
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _submit_word(self):
        start_pos = self.selection_start_pos
        end_pos = tuple(self.cursor_pos)
        self.selection_start_pos = None

        line_cells = self._get_line_cells(start_pos, end_pos)
        selected_word = "".join([self.grid[r, c] for r, c in line_cells])
        
        # Check forward and reverse
        found = False
        for word in [selected_word, selected_word[::-1]]:
            if word in self.solutions and word not in self.found_words:
                sol_start, sol_end = self.solutions[word]
                # Check if submitted line matches solution line
                if (start_pos, end_pos) == (sol_start, sol_end) or \
                   (start_pos, end_pos) == (sol_end, sol_start):
                    self.found_words.add(word)
                    self.found_words_info[word] = {'cells': line_cells}
                    self._spawn_word_found_particles(line_cells)
                    # SFX: Correct word found
                    return 10.0
        
        # SFX: Incorrect submission
        self.incorrect_flash_timer = 15 # frames
        return -1.0

    def _get_line_cells(self, start, end):
        (r1, c1), (r2, c2) = start, end
        dr, dc = abs(r2 - r1), abs(c2 - c1)
        sr = 1 if r2 > r1 else -1
        sc = 1 if c2 > c1 else -1
        
        # Check if line is valid (horizontal, vertical, or 45-degree diagonal)
        if not (r1 == r2 or c1 == c2 or dr == dc):
            return [(r1, c1)] # Return only start point for invalid lines

        cells = []
        err = dr - dc
        r, c = r1, c1
        while True:
            cells.append((r, c))
            if r == r2 and c == c2:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
        return cells

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_ui()
        self._render_particles()

        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        grid_surface = pygame.Surface((self.GRID_SIZE * self.CELL_SIZE, self.GRID_SIZE * self.CELL_SIZE))
        grid_surface.fill(self.COLOR_GRID_BG)
        
        # Draw highlights for found words
        for word, info in self.found_words_info.items():
            for r, c in info['cells']:
                pygame.draw.rect(grid_surface, self.COLOR_FOUND_WORD_BG, 
                                 (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Draw current selection line
        if self.selection_start_pos:
            selection_surface = grid_surface.copy()
            selection_surface.set_colorkey((0,0,0))
            selection_surface.set_alpha(self.COLOR_SELECTION[3])
            
            line_cells = self._get_line_cells(self.selection_start_pos, self.cursor_pos)
            for r, c in line_cells:
                 pygame.draw.rect(selection_surface, self.COLOR_SELECTION, 
                                 (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))
            grid_surface.blit(selection_surface, (0,0))

        # Draw letters and grid lines
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                rect = (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(grid_surface, self.COLOR_GRID_LINES, rect, 1)
                
                letter_surf = self.font_grid.render(self.grid[r, c], True, self.COLOR_TEXT_NORMAL)
                letter_rect = letter_surf.get_rect(center=(rect[0] + self.CELL_SIZE / 2, rect[1] + self.CELL_SIZE / 2))
                grid_surface.blit(letter_surf, letter_rect)

        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = (cursor_c * self.CELL_SIZE, cursor_r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        blink_alpha = 128 + 127 * math.sin(pygame.time.get_ticks() * 0.01)
        pygame.gfxdraw.rectangle(grid_surface, cursor_rect, (*self.COLOR_CURSOR, int(blink_alpha)))
        pygame.draw.rect(grid_surface, self.COLOR_CURSOR, cursor_rect, 2)
        
        # Draw incorrect submission flash
        if self.incorrect_flash_timer > 0:
            flash_alpha = (self.incorrect_flash_timer / 15) * self.COLOR_INCORRECT_FLASH[3]
            flash_surface = pygame.Surface(grid_surface.get_size(), pygame.SRCALPHA)
            flash_surface.fill((*self.COLOR_INCORRECT_FLASH[:3], flash_alpha))
            grid_surface.blit(flash_surface, (0,0))

        self.screen.blit(grid_surface, (self.GRID_OFFSET_X, self.GRID_OFFSET_Y))

    def _render_ui(self):
        ui_x = self.GRID_AREA_WIDTH + 20
        
        # Timer
        timer_text = f"{max(0, int(self.time_remaining // 60)):02}:{max(0, int(self.time_remaining % 60)):02}"
        timer_surf = self.font_ui_timer.render(timer_text, True, self.COLOR_CURSOR)
        self.screen.blit(timer_surf, (ui_x, 20))

        # Score
        score_title_surf = self.font_ui_body.render("Score", True, self.COLOR_TEXT_DIM)
        self.screen.blit(score_title_surf, (self.SCREEN_WIDTH - 80, 20))
        score_surf = self.font_ui_body.render(f"{int(self.score)}", True, self.COLOR_TEXT_NORMAL)
        score_rect = score_surf.get_rect(topright=(self.SCREEN_WIDTH - 20, 40))
        self.screen.blit(score_surf, score_rect)

        # Word List
        words_title_surf = self.font_ui_title.render("Words to Find", True, self.COLOR_TEXT_NORMAL)
        self.screen.blit(words_title_surf, (ui_x, 80))

        y_offset = 110
        col1_x = ui_x
        col2_x = ui_x + 110
        for i, word in enumerate(self.WORD_LIST):
            x = col1_x if i < (len(self.WORD_LIST) + 1) / 2 else col2_x
            y = y_offset + (i % ((len(self.WORD_LIST) + 1) // 2)) * 20
            
            color = self.COLOR_TEXT_SUCCESS if word in self.found_words else self.COLOR_TEXT_NORMAL
            word_surf = self.font_ui_body.render(word, True, color)
            
            if word in self.found_words:
                pygame.draw.line(word_surf, self.COLOR_TEXT_SUCCESS, (0, word_surf.get_height() // 2), (word_surf.get_width(), word_surf.get_height() // 2), 2)

            self.screen.blit(word_surf, (x, y))

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((self.COLOR_BG[0], self.COLOR_BG[1], self.COLOR_BG[2], 220))
        
        win = len(self.found_words) == len(self.WORD_LIST)
        text = "YOU WIN!" if win else "TIME'S UP!"
        color = self.COLOR_TEXT_SUCCESS if win else self.COLOR_CURSOR
        
        text_surf = self.font_game_over.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text_surf, text_rect)

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.1 # Gravity
            p['life'] -= 1

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, p['life'] * 5))
            color = (*p['color'], alpha)
            size = p['size'] * (p['life'] / p['max_life'])
            if size > 0:
                rect = pygame.Rect(p['x'] - size/2, p['y'] - size/2, size, size)
                pygame.gfxdraw.box(self.screen, rect, color)
    
    def _spawn_word_found_particles(self, cells):
        center_r = sum(r for r, c in cells) / len(cells)
        center_c = sum(c for r, c in cells) / len(cells)
        
        center_x = self.GRID_OFFSET_X + center_c * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = self.GRID_OFFSET_Y + center_r * self.CELL_SIZE + self.CELL_SIZE / 2

        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            life = self.np_random.integers(30, 60)
            self.particles.append({
                'x': center_x, 'y': center_y,
                'vx': math.cos(angle) * speed, 'vy': math.sin(angle) * speed,
                'life': life, 'max_life': life,
                'size': self.np_random.uniform(2, 5),
                'color': random.choice(self.PARTICLE_COLORS)
            })

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": self.time_remaining,
            "words_found": len(self.found_words),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        print("✓ Running implementation validation...")
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
        assert "score" in info
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        assert "steps" in info

        # Test termination
        self.time_remaining = 0
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert term == True

        self.reset()
        self.found_words = set(self.WORD_LIST)
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert term == True
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Grid")
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0 # 0: none, 1: up, 2: down, 3: left, 4: right
    space_held = 0 # 0: released, 1: held
    shift_held = 0 # 0: released, 1: held

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        movement = 0
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Words Found: {info['words_found']}")
            
        # Draw the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(GameEnv.FPS)

    env.close()