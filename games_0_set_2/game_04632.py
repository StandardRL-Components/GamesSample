
# Generated: 2025-08-28T02:58:47.627214
# Source Brief: brief_04632.md
# Brief Index: 4632

        
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
    """
    A fast-paced word search game where the player must find hidden words in a grid before time runs out.
    The player controls a cursor to select letters, forming words to score points.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrows to move the cursor. Hold SPACE while moving to select letters. "
        "Release SPACE to submit the word. Press SHIFT to clear your current selection."
    )

    game_description = (
        "Find hidden words within a grid before time runs out in this fast-paced word search challenge."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 14, 10
    CELL_SIZE = 32
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 10
    
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    TARGET_WORDS = 15

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_LINE = (50, 56, 72)
    COLOR_LETTER = (200, 210, 230)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION_BG = (255, 200, 0, 100) # RGBA for transparency
    COLOR_SELECTION_PATH = (255, 220, 80)
    COLOR_CORRECT = (0, 255, 120)
    COLOR_INCORRECT = (255, 80, 80)
    COLOR_FOUND_LETTER = (100, 110, 130)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_TIMER_WARN = (255, 100, 100)

    # --- Word List ---
    WORD_BANK = [
        "PYTHON", "AGENT", "REWARD", "ACTION", "SPACE", "POLICY", "LEARN", "GRID", "GAME",
        "STATE", "MODEL", "TENSOR", "BATCH", "EPOCH", "GYM", "STEP", "RESET", "FRAME", "PIXEL",
        "VECTOR", "MATRIX", "KERNEL", "LAYER", "NODE", "GRAPH", "DATA", "CODE", "ALGO", "LOOP",
        "FLOAT", "INTEGER", "SYSTEM", "CLOUD", "SERVER", "CLIENT", "CACHE", "MEMORY", "CPU",
        "GPU", "DRIVE", "FILE", "PATH", "BUG", "DEBUG", "TEST", "BUILD", "DEPLOY", "API"
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
        
        self.font_grid = pygame.font.Font(None, 28)
        self.font_ui = pygame.font.Font(None, 24)
        self.font_ui_large = pygame.font.Font(None, 36)
        
        # State variables initialized in reset()
        self.grid = None
        self.placed_words = None
        self.cursor_pos = None
        self.current_selection_coords = None
        self.found_words = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.time_left = None
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles = []
        self.feedback_flash = 0
        self.feedback_color = self.COLOR_INCORRECT

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_STEPS
        
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.current_selection_coords = []
        self.found_words = set()
        
        self._generate_grid()
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles = []
        self.feedback_flash = 0
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.time_left -= 1
        reward = 0

        # Unpack actions
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle cursor movement
        prev_cursor_pos = list(self.cursor_pos)
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)
        cursor_moved = prev_cursor_pos != self.cursor_pos

        # Handle selection (holding space)
        if space_held:
            current_coord = tuple(self.cursor_pos)
            if not self.current_selection_coords:
                self.current_selection_coords.append(current_coord)
                # Sound effect: selection_start
            elif cursor_moved and current_coord not in self.current_selection_coords:
                last_coord = self.current_selection_coords[-1]
                if self._is_adjacent(current_coord, last_coord):
                    self.current_selection_coords.append(current_coord)
                    reward += 0.01 # Small reward for extending selection
                    # Sound effect: selection_tick
                else:
                    # Invalid move, penalize and clear
                    reward -= 0.1
                    self.current_selection_coords = []
                    self.feedback_flash = 10 # Flash red
                    self.feedback_color = self.COLOR_INCORRECT
        
        # Handle submission (releasing space)
        if self.prev_space_held and not space_held and self.current_selection_coords:
            submission_reward = self._handle_submission()
            reward += submission_reward
        
        # Handle clearing selection (pressing shift)
        if shift_held and not self.prev_shift_held:
            if self.current_selection_coords:
                reward -= 0.05 # Small penalty for clearing
                self.current_selection_coords = []
                # Sound effect: selection_clear

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # Update feedback effects
        if self.feedback_flash > 0:
            self.feedback_flash -= 1
        self._update_particles()
        
        # Check for termination conditions
        terminated = False
        if len(self.found_words) >= self.TARGET_WORDS:
            reward += 10  # Victory bonus
            terminated = True
            # Sound effect: game_win
        elif self.time_left <= 0 or self.steps >= self.MAX_STEPS:
            reward -= 10 # Time out penalty
            terminated = True
            # Sound effect: game_over
        
        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_submission(self):
        word = "".join([self.grid[y][x] for x, y in self.current_selection_coords])
        
        if word in self.placed_words and word not in self.found_words:
            self.found_words.add(word)
            self.feedback_flash = 10
            self.feedback_color = self.COLOR_CORRECT
            
            # Create particles for found word
            for x, y in self.current_selection_coords:
                px = self.GRID_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
                py = self.GRID_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                self._create_particles(px, py)

            self.current_selection_coords = []
            # Sound effect: correct_word
            return len(word) # Reward based on word length
        else:
            self.feedback_flash = 10
            self.feedback_color = self.COLOR_INCORRECT
            self.current_selection_coords = []
            # Sound effect: incorrect_word
            return -1 # Penalty for wrong submission

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.placed_words = {}
        
        words_to_place = random.sample(self.WORD_BANK, k=min(len(self.WORD_BANK), 30))
        
        for word in words_to_place:
            if len(self.placed_words) >= self.TARGET_WORDS:
                break
            
            word = word.upper()
            attempts = 0
            placed = False
            while not placed and attempts < 100:
                attempts += 1
                direction = random.choice([(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)])
                
                max_x = self.GRID_COLS - 1 - (len(word) - 1) * abs(direction[0])
                max_y = self.GRID_ROWS - 1 - (len(word) - 1) * abs(direction[1])
                if direction[1] < 0:
                    start_y = random.randint(len(word)-1, self.GRID_ROWS-1)
                else:
                    start_y = random.randint(0, max_y)
                if direction[0] < 0:
                    start_x = random.randint(len(word)-1, self.GRID_COLS-1)
                else:
                    start_x = random.randint(0, max_x)
                
                if start_x < 0 or start_y < 0: continue

                path = []
                can_place = True
                for i in range(len(word)):
                    x = start_x + i * direction[0]
                    y = start_y + i * direction[1]
                    path.append((x, y))
                    if not (0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS):
                        can_place = False
                        break
                    if self.grid[y][x] != '' and self.grid[y][x] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i, (x, y) in enumerate(path):
                        self.grid[y][x] = word[i]
                    self.placed_words[word] = path
                    placed = True
        
        # Fill empty cells with random letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == '':
                    self.grid[r][c] = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw found words paths (dimmed)
        for word in self.found_words:
            path = self.placed_words[word]
            for x, y in path:
                rect = pygame.Rect(
                    self.GRID_X + x * self.CELL_SIZE,
                    self.GRID_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_FOUND_LETTER, rect, border_radius=4)
        
        # Draw selection highlight
        if self.current_selection_coords:
            for i, (x, y) in enumerate(self.current_selection_coords):
                center_x = self.GRID_X + x * self.CELL_SIZE + self.CELL_SIZE // 2
                center_y = self.GRID_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                
                # Draw translucent circle background
                s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                flash_alpha = 150 if self.feedback_flash > 0 and self.feedback_color == self.COLOR_INCORRECT else self.COLOR_SELECTION_BG[3]
                flash_color = self.feedback_color if self.feedback_flash > 0 else self.COLOR_SELECTION_BG
                pygame.draw.circle(s, (*flash_color[:3], flash_alpha), (self.CELL_SIZE // 2, self.CELL_SIZE // 2), self.CELL_SIZE // 2 - 2)
                self.screen.blit(s, (center_x - self.CELL_SIZE // 2, center_y - self.CELL_SIZE // 2))

                # Draw connecting lines
                if i > 0:
                    prev_x, prev_y = self.current_selection_coords[i-1]
                    prev_center_x = self.GRID_X + prev_x * self.CELL_SIZE + self.CELL_SIZE // 2
                    prev_center_y = self.GRID_Y + prev_y * self.CELL_SIZE + self.CELL_SIZE // 2
                    pygame.draw.line(self.screen, self.COLOR_SELECTION_PATH, (prev_center_x, prev_center_y), (center_x, center_y), 4)

        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + r * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X + c * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), 1)
            
        # Draw letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                letter = self.grid[r][c]
                color = self.COLOR_LETTER
                is_found = any((c, r) in self.placed_words[w] for w in self.found_words)
                if is_found:
                    color = self.COLOR_FOUND_LETTER

                text_surf = self.font_grid.render(letter, True, color)
                text_rect = text_surf.get_rect(center=(
                    self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                ))
                self.screen.blit(text_surf, text_rect)
        
        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_X + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_Y + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2, border_radius=4)
        
        # Draw particles
        for p in self.particles:
            p['life'] -= 1
            p['y'] -= p['vy']
            p['size'] = max(0, p['size'] * 0.95)
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            pygame.gfxdraw.filled_circle(self.screen, int(p['x']), int(p['y']), int(p['size']), (*self.COLOR_CORRECT, alpha))

        # Draw feedback flash for correct submission
        if self.feedback_flash > 0 and self.feedback_color == self.COLOR_CORRECT:
            s = pygame.Surface((self.GRID_WIDTH, self.GRID_HEIGHT), pygame.SRCALPHA)
            alpha = 128 * (self.feedback_flash / 10)
            s.fill((*self.COLOR_CORRECT, alpha))
            self.screen.blit(s, (self.GRID_X, self.GRID_Y))

    def _render_ui(self):
        # Words Found
        found_text = f"FOUND: {len(self.found_words)} / {self.TARGET_WORDS}"
        self._render_text(found_text, self.font_ui_large, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH - 120, 30), "topright")

        # Time
        time_sec = self.time_left / self.FPS
        time_color = self.COLOR_UI_TEXT
        if time_sec < 10 and self.steps % self.FPS < self.FPS / 2: # Flashing effect
            time_color = self.COLOR_TIMER_WARN
        time_text = f"TIME: {time_sec:.1f}"
        self._render_text(time_text, self.font_ui_large, time_color, (120, 30), "topleft")

        # Current Selection
        current_word = "".join([self.grid[y][x] for x, y in self.current_selection_coords])
        selection_text = f"SELECTING: {current_word}"
        self._render_text(selection_text, self.font_ui, self.COLOR_UI_TEXT, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 20), "midbottom")

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_found": len(self.found_words),
            "time_left_seconds": self.time_left / self.FPS,
        }

    def _render_text(self, text, font, color, pos, anchor="center"):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        setattr(text_rect, anchor, pos)
        self.screen.blit(text_surf, text_rect)

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1

    def _create_particles(self, x, y):
        for _ in range(5):
            life = self.FPS * (0.5 + random.random() * 0.5)
            self.particles.append({
                'x': x + random.uniform(-5, 5),
                'y': y + random.uniform(-5, 5),
                'vx': random.uniform(-0.5, 0.5),
                'vy': random.uniform(0.5, 1.5),
                'size': random.uniform(2, 5),
                'life': life,
                'max_life': life,
            })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
        assert len(self.placed_words) >= self.TARGET_WORDS
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be executed when the environment is used by Gymnasium
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Search Challenge")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Words Found: {info['words_found']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(GameEnv.FPS)

    pygame.quit()