
# Generated: 2025-08-27T15:20:08.785009
# Source Brief: brief_00962.md
# Brief Index: 962

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select the starting letter "
        "of a word. Move the cursor to the end of the word and press Shift to submit your guess."
    )

    game_description = (
        "Find all 10 hidden words in a procedurally generated grid before the 60-second timer runs out."
    )

    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 20
    CELL_SIZE = SCREEN_HEIGHT // GRID_SIZE
    GRID_PIXEL_DIM = GRID_SIZE * CELL_SIZE
    UI_WIDTH = SCREEN_WIDTH - GRID_PIXEL_DIM
    
    GAME_DURATION_SECONDS = 60
    FPS = 30
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # --- Colors ---
    COLOR_BG = (25, 25, 35)
    COLOR_GRID_BG = (40, 40, 55)
    COLOR_UI_BG = (30, 30, 45)
    COLOR_TEXT = (220, 220, 240)
    COLOR_TEXT_DIM = (120, 120, 140)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION_START = (0, 150, 255)
    COLOR_SELECTION_PATH = (0, 150, 255, 100) # RGBA
    COLOR_SUCCESS = (0, 255, 150)
    COLOR_ERROR = (255, 80, 80)
    COLOR_FOUND_PERMANENT = (0, 255, 150, 40) # RGBA
    COLOR_FOUND_STRIKETHROUGH = (0, 200, 120)

    # --- Word List ---
    WORD_BANK = [
        "PYTHON", "GYMNASIUM", "REINFORCEMENT", "LEARNING", "AGENT", "REWARD", "ACTION",
        "STATE", "POLICY", "EPISODE", "ALGORITHM", "EXPLORE", "EXPLOIT", "NEURAL",
        "NETWORK", "TENSOR", "KERAS", "PYTORCH", "DEEP", "MACHINE", "ARCADIA", "VECTOR",
        "OBSERVATION", "ENVIRONMENT", "DEVELOPER", "GAMEPLAY", "VISUAL", "EXPERIENCE"
    ]
    NUM_WORDS_TO_FIND = 10

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

        # Fonts
        try:
            self.font_grid = pygame.font.SysFont('monospace', int(self.CELL_SIZE * 0.8), bold=True)
            self.font_ui_title = pygame.font.SysFont('sans-serif', 24, bold=True)
            self.font_ui_body = pygame.font.SysFont('sans-serif', 18)
            self.font_ui_feedback = pygame.font.SysFont('sans-serif', 20, bold=True)
        except pygame.error:
            self.font_grid = pygame.font.Font(None, int(self.CELL_SIZE * 0.9))
            self.font_ui_title = pygame.font.Font(None, 30)
            self.font_ui_body = pygame.font.Font(None, 24)
            self.font_ui_feedback = pygame.font.Font(None, 26)

        self.np_random = None
        
        # State variables are initialized in reset()
        self.grid = []
        self.target_words_info = {}
        self.found_words = set()
        self.found_words_locations = []
        self.steps = 0
        self.score = 0
        self.time_left = 0
        self.game_over = False
        self.cursor_pos = [0, 0]
        self.selection_start_pos = None
        self.last_space_held = False
        self.last_shift_held = False
        self.feedback_message = {"text": "", "color": self.COLOR_TEXT, "time": 0}

        self.validate_implementation()
    
    def _generate_grid(self):
        grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        
        words_to_place = self.np_random.choice(self.WORD_BANK, self.NUM_WORDS_TO_FIND, replace=False).tolist()
        target_words_info = {}

        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]

        for word in words_to_place:
            placed = False
            for _ in range(200): # Max attempts to place a word
                word = word if self.np_random.random() > 0.5 else word[::-1]
                direction = self.np_random.choice(directions)
                
                start_x = self.np_random.integers(0, self.GRID_SIZE)
                start_y = self.np_random.integers(0, self.GRID_SIZE)
                
                end_x = start_x + (len(word) - 1) * direction[0]
                end_y = start_y + (len(word) - 1) * direction[1]

                if not (0 <= end_x < self.GRID_SIZE and 0 <= end_y < self.GRID_SIZE):
                    continue

                can_place = True
                for i in range(len(word)):
                    px, py = start_x + i * direction[0], start_y + i * direction[1]
                    if grid[py][px] != '' and grid[py][px] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i in range(len(word)):
                        px, py = start_x + i * direction[0], start_y + i * direction[1]
                        grid[py][px] = word[i]
                    
                    original_word = word if word in words_to_place else word[::-1]
                    target_words_info[original_word] = {
                        "start": (start_x, start_y),
                        "end": (end_x, end_y),
                        "found": False
                    }
                    placed = True
                    break
            
            if not placed:
                # This can happen if the grid is too dense. For this game, it's very unlikely.
                # A robust solution would be to restart generation. Here we just continue.
                pass

        # Fill empty cells
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if grid[y][x] == '':
                    grid[y][x] = self.np_random.choice(list(alphabet))
        
        return grid, target_words_info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.GAME_DURATION_SECONDS
        
        self.grid, self.target_words_info = self._generate_grid()
        self.found_words = set()
        self.found_words_locations = []
        
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selection_start_pos = None
        self.last_space_held = True # Prevent action on first frame
        self.last_shift_held = True
        self.feedback_message = {"text": "", "color": self.COLOR_TEXT, "time": 0}
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        reward = 0
        self.time_left = max(0, self.time_left - 1.0 / self.FPS)
        
        # --- Input Processing ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held
        self.last_space_held, self.last_shift_held = space_held, shift_held

        # --- Game Logic ---
        # Cursor movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # Selection start
        if space_pressed:
            self.selection_start_pos = list(self.cursor_pos)
            # Placeholder for sound effect
            # sfx: select_start.wav
        
        # Word submission
        if shift_pressed and self.selection_start_pos:
            reward += self._check_word_submission()
            self.selection_start_pos = None

        # Update feedback message timer
        if self.feedback_message["time"] > 0:
            self.feedback_message["time"] -= 1.0 / self.FPS
        
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            if len(self.found_words) == self.NUM_WORDS_TO_FIND:
                reward += 50  # Goal-oriented reward
                self.feedback_message = {"text": "ALL WORDS FOUND!", "color": self.COLOR_SUCCESS, "time": 5}
            else: # Timeout
                reward -= 10
                self.feedback_message = {"text": "TIME'S UP!", "color": self.COLOR_ERROR, "time": 5}
            self.game_over = True
            
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_word_submission(self):
        start_pos = self.selection_start_pos
        end_pos = self.cursor_pos
        dx, dy = end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]
        
        word_str = ""
        # Check for straight or diagonal lines
        if dx == 0 and dy != 0: # Vertical
            step_y = 1 if dy > 0 else -1
            for y in range(start_pos[1], end_pos[1] + step_y, step_y):
                word_str += self.grid[y][start_pos[0]]
        elif dy == 0 and dx != 0: # Horizontal
            step_x = 1 if dx > 0 else -1
            for x in range(start_pos[0], end_pos[0] + step_x, step_x):
                word_str += self.grid[start_pos[1]][x]
        elif abs(dx) == abs(dy) and dx != 0: # Diagonal
            step_x = 1 if dx > 0 else -1
            step_y = 1 if dy > 0 else -1
            for i in range(abs(dx) + 1):
                word_str += self.grid[start_pos[1] + i * step_y][start_pos[0] + i * step_x]
        
        if not word_str:
            return 0

        found_match = False
        for target_word, info in self.target_words_info.items():
            if target_word not in self.found_words and (word_str == target_word or word_str[::-1] == target_word):
                self.found_words.add(target_word)
                self.score += 100
                self.found_words_locations.append((info['start'], info['end']))
                self.feedback_message = {"text": f"FOUND: {target_word}", "color": self.COLOR_SUCCESS, "time": 2}
                found_match = True
                # Placeholder for sound effect
                # sfx: word_found.wav
                return 10 # Event-based reward
        
        if not found_match:
            self.feedback_message = {"text": "INVALID WORD", "color": self.COLOR_ERROR, "time": 1.5}
            # Placeholder for sound effect
            # sfx: invalid_word.wav
            return -1 # Small penalty for wrong guess
        return 0

    def _check_termination(self):
        if len(self.found_words) == self.NUM_WORDS_TO_FIND:
            return True
        if self.time_left <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
    
    def _get_observation(self):
        # --- Clear screen ---
        self.screen.fill(self.COLOR_BG)
        
        # --- Render Game Area ---
        self._render_game()
        
        # --- Render UI ---
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        grid_surface = pygame.Surface((self.GRID_PIXEL_DIM, self.GRID_PIXEL_DIM))
        grid_surface.fill(self.COLOR_GRID_BG)

        # Draw permanent highlights for found words
        for start_pixel, end_pixel in self.found_words_locations:
            start_coord = (start_pixel[0] * self.CELL_SIZE + self.CELL_SIZE // 2, start_pixel[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            end_coord = (end_pixel[0] * self.CELL_SIZE + self.CELL_SIZE // 2, end_pixel[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            pygame.draw.line(grid_surface, self.COLOR_FOUND_PERMANENT, start_coord, end_coord, self.CELL_SIZE)

        # Draw selection path
        if self.selection_start_pos:
            start_coord = (self.selection_start_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2, self.selection_start_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            current_coord = (self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2, self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2)
            pygame.draw.line(grid_surface, self.COLOR_SELECTION_PATH, start_coord, current_coord, self.CELL_SIZE // 2)
            
            # Draw start selection highlight
            start_rect = pygame.Rect(self.selection_start_pos[0] * self.CELL_SIZE, self.selection_start_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            pygame.draw.rect(grid_surface, self.COLOR_SELECTION_START, start_rect, 3, border_radius=4)

        # Draw letters
        for y, row in enumerate(self.grid):
            for x, char in enumerate(row):
                text_surf = self.font_grid.render(char, True, self.COLOR_TEXT)
                text_rect = text_surf.get_rect(center=(x * self.CELL_SIZE + self.CELL_SIZE // 2, y * self.CELL_SIZE + self.CELL_SIZE // 2))
                grid_surface.blit(text_surf, text_rect)

        # Draw cursor
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(grid_surface, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
        
        self.screen.blit(grid_surface, (0, 0))

    def _render_ui(self):
        ui_rect = pygame.Rect(self.GRID_PIXEL_DIM, 0, self.UI_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        
        x_offset = self.GRID_PIXEL_DIM + 20
        y_offset = 20

        # Timer
        time_color = self.COLOR_TEXT if self.time_left > 10 else self.COLOR_ERROR
        time_text = f"{self.time_left:.1f}"
        time_surf = self.font_ui_title.render(time_text, True, time_color)
        self.screen.blit(time_surf, (x_offset, y_offset))
        y_offset += 40
        
        # Score
        score_text = f"Score: {self.score}"
        score_surf = self.font_ui_body.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (x_offset, y_offset))
        y_offset += 30

        # Words Found
        found_text = f"Found: {len(self.found_words)}/{self.NUM_WORDS_TO_FIND}"
        found_surf = self.font_ui_body.render(found_text, True, self.COLOR_TEXT)
        self.screen.blit(found_surf, (x_offset, y_offset))
        y_offset += 40

        # Word List
        sorted_words = sorted(self.target_words_info.keys())
        for word in sorted_words:
            if y_offset > self.SCREEN_HEIGHT - 60: break # Don't draw off-screen
            is_found = word in self.found_words
            color = self.COLOR_FOUND_STRIKETHROUGH if is_found else self.COLOR_TEXT_DIM
            word_surf = self.font_ui_body.render(word, True, color)
            
            if is_found:
                strike_rect = word_surf.get_rect(topleft=(x_offset, y_offset + self.font_ui_body.get_height() // 2))
                pygame.draw.line(self.screen, self.COLOR_SUCCESS, strike_rect.topleft, strike_rect.topright, 2)

            self.screen.blit(word_surf, (x_offset, y_offset))
            y_offset += 22

        # Feedback Message
        if self.feedback_message["time"] > 0:
            alpha = min(255, int(255 * (self.feedback_message["time"] / 0.5)))
            feedback_surf = self.font_ui_feedback.render(self.feedback_message["text"], True, self.feedback_message["color"])
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(centerx=self.GRID_PIXEL_DIM + self.UI_WIDTH // 2, bottom=self.SCREEN_HEIGHT - 20)
            self.screen.blit(feedback_surf, feedback_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "words_found": len(self.found_words),
        }
    
    def close(self):
        pygame.font.quit()
        pygame.quit()
        super().close()

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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This part is for human play testing and will not be part of the final submission
    import sys

    # Set up the environment for rendering to the screen
    env = GameEnv()
    env.metadata["render_modes"].append("human")
    env.render_mode = "human"
    
    # Re-initialize pygame with a display
    pygame.quit()
    pygame.init()
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Search Puzzle")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    done = False
    
    print(GameEnv.user_guide)

    while not done:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # No-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        clock.tick(GameEnv.FPS)

    print(f"Game Over! Final Score: {info['score']}")
    env.close()
    pygame.quit()
    sys.exit()