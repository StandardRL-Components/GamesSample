
# Generated: 2025-08-27T15:21:13.222415
# Source Brief: brief_00964.md
# Brief Index: 964

        
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
        "Controls: Use arrow keys to move the cursor. Press and hold space to start selecting a word, "
        "drag to the end of the word, and release space to submit. Press shift to cancel a selection."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A classic word search puzzle. Find all the hidden words in the grid before the 60-second timer runs out. "
        "Found words are highlighted on the grid and struck through on the list."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_AREA_WIDTH = 400
    UI_AREA_WIDTH = SCREEN_WIDTH - GRID_AREA_WIDTH
    GRID_COLS, GRID_ROWS = 20, 15
    CELL_WIDTH = GRID_AREA_WIDTH // GRID_COLS
    CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS
    GAME_DURATION_SECONDS = 60
    FPS = 30
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_GRID_BG = (30, 35, 50)
    COLOR_UI_BG = (40, 45, 60)
    COLOR_LETTER = (200, 210, 230)
    COLOR_FOUND_LETTER = (80, 90, 110)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION_FILL = (255, 200, 0, 100)
    COLOR_SELECTION_LINE = (255, 255, 0)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEADER = (255, 200, 0)
    COLOR_STRIKETHROUGH = (255, 80, 80)
    COLOR_FLASH_CORRECT = (0, 255, 100, 150)
    COLOR_FLASH_INCORRECT = (255, 50, 50, 150)
    
    WORD_LIST = [
        "PYTHON", "GYMNASIUM", "REWARD", "ACTION", "AGENT", "POLICY", 
        "STATE", "EPISODE", "VECTOR", "TENSOR", "LEARNING", "SEARCH", 
        "GRID", "PUZZLE", "TIMER", "SCORE", "VISUAL", "EXPERT", "GAME", "SOLVE"
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
        
        self.letter_font = pygame.font.Font(None, 20)
        self.ui_font = pygame.font.Font(None, 22)
        self.ui_header_font = pygame.font.Font(None, 28)
        self.game_over_font = pygame.font.Font(None, 72)
        
        self.grid = []
        self.word_locations = {}
        self.cursor_pos = [0, 0]
        self.selection_start = None
        self.found_words = set()
        self.found_word_cells = set()
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = 0
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        self.flash_alpha = 0
        self.flash_color = (0,0,0,0)

        self.particles = []

        self.reset()
        
        try:
            self.validate_implementation()
        except AssertionError as e:
            print(f"Implementation validation failed: {e}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selection_start = None
        self.found_words = set()
        self.found_word_cells = set()
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.flash_alpha = 0
        self.particles = []
        
        self._generate_grid()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        
        if not self.game_over:
            self.time_remaining -= 1
            self.steps += 1
            
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            reward += self._handle_input(movement, space_held, shift_held)
            
            self.prev_space_held = space_held
            self.prev_shift_held = shift_held

        self._update_effects()

        terminated = self._check_termination()
        if terminated and not self.game_over: # First frame of termination
            self.game_over = True
            if len(self.found_words) == len(self.WORD_LIST):
                reward += 50 # Win bonus
            elif self.time_remaining <= 0:
                reward -= 50 # Time out penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        
        # --- Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_ROWS - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_COLS - 1, self.cursor_pos[0] + 1)

        # --- Shift to Cancel ---
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.selection_start:
            self.selection_start = None
            # sfx: cancel_sound

        # --- Space to Select/Confirm ---
        if space_held and not self.prev_space_held and not self.selection_start:
            self.selection_start = tuple(self.cursor_pos)
            # sfx: selection_start_sound
        
        space_released = not space_held and self.prev_space_held
        if space_released and self.selection_start:
            selection_end = tuple(self.cursor_pos)
            reward += self._check_selection(self.selection_start, selection_end)
            self.selection_start = None
            
        return reward

    def _check_selection(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        word_str = ""
        coords = []

        if dx == 0 and dy == 0: # Single letter selection
            return 0

        if dx == 0 or dy == 0 or abs(dx) == abs(dy): # Valid line (horizontal, vertical, diagonal)
            length = max(abs(dx), abs(dy))
            step_x = dx // length if dx != 0 else 0
            step_y = dy // length if dy != 0 else 0
            
            for i in range(length + 1):
                x = start[0] + i * step_x
                y = start[1] + i * step_y
                word_str += self.grid[y][x]
                coords.append((x, y))

            if (word_str in self.WORD_LIST and word_str not in self.found_words) or \
               (word_str[::-1] in self.WORD_LIST and word_str[::-1] not in self.found_words):
                
                actual_word = word_str if word_str in self.WORD_LIST else word_str[::-1]
                self.found_words.add(actual_word)
                self.found_word_cells.update(coords)
                
                self.score += 10
                self._trigger_flash(self.COLOR_FLASH_CORRECT)
                self._create_particles_for_word(coords)
                # sfx: word_found_sound
                return 10
        
        # Incorrect guess
        self._trigger_flash(self.COLOR_FLASH_INCORRECT)
        # sfx: incorrect_guess_sound
        return -0.1 # Small penalty for wrong guess

    def _check_termination(self):
        if self.time_remaining <= 0:
            return True
        if len(self.found_words) == len(self.WORD_LIST):
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.word_locations = {}
        
        words_to_place = sorted(self.WORD_LIST, key=len, reverse=True)

        for word in words_to_place:
            placed = False
            for _ in range(200): # Placement attempts
                direction = self.np_random.choice([
                    (1, 0), (0, 1), (1, 1), (1, -1),
                    (-1, 0), (0, -1), (-1, -1), (-1, 1)
                ])
                if self.np_random.random() > 0.5: # Reverse word
                    word_to_place = word[::-1]
                else:
                    word_to_place = word
                
                start_x = self.np_random.integers(0, self.GRID_COLS)
                start_y = self.np_random.integers(0, self.GRID_ROWS)

                end_x = start_x + (len(word_to_place) - 1) * direction[0]
                end_y = start_y + (len(word_to_place) - 1) * direction[1]

                if 0 <= end_x < self.GRID_COLS and 0 <= end_y < self.GRID_ROWS:
                    # Check for overlaps
                    can_place = True
                    coords = []
                    for i in range(len(word_to_place)):
                        x = start_x + i * direction[0]
                        y = start_y + i * direction[1]
                        if self.grid[y][x] != '':
                            can_place = False
                            break
                        coords.append((x,y))
                    
                    if can_place:
                        for i, char in enumerate(word_to_place):
                            x, y = coords[i]
                            self.grid[y][x] = char
                        self.word_locations[word] = coords
                        placed = True
                        break
            if not placed:
                # This can happen if the grid is too small or words are too long.
                # For this setup, it's very unlikely. We'll proceed with a potentially incomplete grid.
                # A more robust solution would be to restart generation.
                print(f"Warning: Could not place word '{word}'")

        # Fill empty spaces
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == '':
                    self.grid[r][c] = chr(self.np_random.integers(65, 91)) # A-Z

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        self._render_effects()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, (0, 0, self.GRID_AREA_WIDTH, self.SCREEN_HEIGHT))
        
        # Draw letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color = self.COLOR_LETTER
                if (c, r) in self.found_word_cells:
                    color = self.COLOR_FOUND_LETTER
                
                letter_surf = self.letter_font.render(self.grid[r][c], True, color)
                letter_rect = letter_surf.get_rect(center=(
                    c * self.CELL_WIDTH + self.CELL_WIDTH // 2,
                    r * self.CELL_HEIGHT + self.CELL_HEIGHT // 2
                ))
                self.screen.blit(letter_surf, letter_rect)

        # Draw selection highlight
        if self.selection_start:
            start_pos = self.selection_start
            end_pos = self.cursor_pos
            
            # Draw line connecting selection points
            start_px = (start_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH // 2, start_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT // 2)
            end_px = (end_pos[0] * self.CELL_WIDTH + self.CELL_WIDTH // 2, end_pos[1] * self.CELL_HEIGHT + self.CELL_HEIGHT // 2)
            pygame.draw.line(self.screen, self.COLOR_SELECTION_LINE, start_px, end_px, 2)
            pygame.gfxdraw.aacircle(self.screen, start_px[0], start_px[1], 5, self.COLOR_SELECTION_LINE)
            pygame.gfxdraw.filled_circle(self.screen, start_px[0], start_px[1], 5, self.COLOR_SELECTION_LINE)


        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            cx * self.CELL_WIDTH, cy * self.CELL_HEIGHT,
            self.CELL_WIDTH, self.CELL_HEIGHT
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=3)

    def _render_ui(self):
        ui_x = self.GRID_AREA_WIDTH
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x, 0, self.UI_AREA_WIDTH, self.SCREEN_HEIGHT))
        
        # --- Top Info Bar ---
        time_str = f"TIME: {max(0, math.ceil(self.time_remaining / self.FPS))}"
        score_str = f"SCORE: {self.score}"
        found_str = f"FOUND: {len(self.found_words)}/{len(self.WORD_LIST)}"

        time_surf = self.ui_font.render(time_str, True, self.COLOR_UI_TEXT)
        score_surf = self.ui_font.render(score_str, True, self.COLOR_UI_TEXT)
        found_surf = self.ui_font.render(found_str, True, self.COLOR_UI_TEXT)

        self.screen.blit(time_surf, (ui_x + 10, 10))
        self.screen.blit(score_surf, (ui_x + 10, 35))
        self.screen.blit(found_surf, (ui_x + 10, 60))

        # --- Word List ---
        header_surf = self.ui_header_font.render("WORDS TO FIND", True, self.COLOR_UI_HEADER)
        self.screen.blit(header_surf, (ui_x + 10, 100))

        y_offset = 135
        for i, word in enumerate(self.WORD_LIST):
            color = self.COLOR_UI_TEXT
            is_found = word in self.found_words
            if is_found:
                color = self.COLOR_FOUND_LETTER

            word_surf = self.ui_font.render(word, True, color)
            pos = (ui_x + 20, y_offset + i * 20)
            self.screen.blit(word_surf, pos)
            
            if is_found:
                pygame.draw.line(self.screen, self.COLOR_STRIKETHROUGH, 
                                 (pos[0] - 2, pos[1] + word_surf.get_height() // 2), 
                                 (pos[0] + word_surf.get_width() + 2, pos[1] + word_surf.get_height() // 2), 2)

    def _render_effects(self):
        # Flash effect
        if self.flash_alpha > 0:
            self.flash_surface.fill(self.flash_color)
            self.flash_surface.set_alpha(self.flash_alpha)
            self.screen.blit(self.flash_surface, (0, 0))
        
        # Particles
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), p['color'])

    def _update_effects(self):
        # Flash
        if self.flash_alpha > 0:
            self.flash_alpha = max(0, self.flash_alpha - 15)
        
        # Particles
        new_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] *= 0.98
            if p['life'] > 0 and p['radius'] > 0.5:
                new_particles.append(p)
        self.particles = new_particles

    def _trigger_flash(self, color):
        self.flash_color = color
        self.flash_alpha = color[3]

    def _create_particles_for_word(self, coords):
        if not coords: return
        
        # Find center of the word
        xs, ys = zip(*coords)
        center_x = (min(xs) + max(xs)) / 2 * self.CELL_WIDTH + self.CELL_WIDTH / 2
        center_y = (min(ys) + max(ys)) / 2 * self.CELL_HEIGHT + self.CELL_HEIGHT / 2
        
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'radius': random.uniform(3, 7),
                'life': random.randint(20, 40),
                'color': (
                    random.randint(150, 255),
                    random.randint(200, 255),
                    random.randint(100, 200),
                    200
                )
            })

    def _render_game_over(self):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        if len(self.found_words) == len(self.WORD_LIST):
            text = "YOU WIN!"
            color = (100, 255, 100)
        else:
            text = "TIME'S UP!"
            color = (255, 100, 100)
        
        text_surf = self.game_over_font.render(text, True, color)
        text_rect = text_surf.get_rect(center=self.screen.get_rect().center)
        self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining": max(0, math.ceil(self.time_remaining / self.FPS)),
            "words_found": len(self.found_words),
        }

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    
    # Use a dictionary to track held keys for smoother controls
    keys_held = {
        pygame.K_UP: False, pygame.K_DOWN: False,
        pygame.K_LEFT: False, pygame.K_RIGHT: False,
        pygame.K_SPACE: False, pygame.K_LSHIFT: False, pygame.K_RSHIFT: False
    }

    # We need a window to capture keyboard events
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Search")
    
    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in keys_held:
                    keys_held[event.key] = True
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
            if event.type == pygame.KEYUP:
                if event.key in keys_held:
                    keys_held[event.key] = False

        # --- Action Mapping ---
        movement = 0 # none
        if keys_held[pygame.K_UP]: movement = 1
        elif keys_held[pygame.K_DOWN]: movement = 2
        elif keys_held[pygame.K_LEFT]: movement = 3
        elif keys_held[pygame.K_RIGHT]: movement = 4
        
        space = 1 if keys_held[pygame.K_SPACE] else 0
        shift = 1 if keys_held[pygame.K_LSHIFT] or keys_held[pygame.K_RSHIFT] else 0

        action = [movement, space, shift]
        
        # --- Step Environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Render ---
        # The observation is already the rendered frame, so we just need to display it
        # Pygame uses (width, height), numpy uses (height, width), so we need to transpose
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait a bit before allowing a reset
            pygame.time.wait(2000)
            obs, info = env.reset()

    env.close()