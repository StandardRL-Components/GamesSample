
# Generated: 2025-08-28T04:24:27.995664
# Source Brief: brief_02307.md
# Brief Index: 2307

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import time
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press and hold space to select letters. "
        "Release space to cancel. Press shift to submit the selected word."
    )

    game_description = (
        "A fast-paced word search puzzle. Find all the hidden words in the grid before the timer runs out. "
        "Score points for each word found and a bonus for completing the puzzle."
    )

    auto_advance = True

    # --- Constants ---
    # Visuals
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 18, 12
    CELL_SIZE = 24
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 4
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2 + 20

    # Colors
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_BG = (35, 38, 46)
    COLOR_GRID_LINE = (50, 53, 61)
    COLOR_TEXT = (220, 220, 220)
    COLOR_CURSOR = (255, 190, 0)
    COLOR_SELECTION = (255, 190, 0, 100)
    COLOR_CORRECT = (0, 255, 120, 120)
    COLOR_FOUND_WORD = (80, 80, 80)
    COLOR_FOUND_STRIKE = (120, 120, 120)
    COLOR_UI_BG = (45, 48, 56)
    COLOR_TIMER_LOW = (255, 80, 80)
    COLOR_MSG_CORRECT = (0, 255, 120)
    COLOR_MSG_INCORRECT = (255, 80, 80)
    COLOR_MSG_DUPE = (255, 190, 0)

    # Gameplay
    TOTAL_TIME = 60.0  # seconds
    MAX_STEPS = 1800 # 60s at 30fps
    WORD_LIST = [
        "PYTHON", "AGENT", "REWARD", "ACTION", "STATE", "POLICY", "LEARN",
        "GRID", "VECTOR", "TENSOR", "GYM", "SEARCH", "PUZZLE", "SOLVE", "CODE"
    ]
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_grid = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_ui = pygame.font.SysFont("Segoe UI", 16)
            self.font_ui_bold = pygame.font.SysFont("Segoe UI", 18, bold=True)
            self.font_msg = pygame.font.SysFont("Segoe UI", 32, bold=True)
        except pygame.error:
            self.font_grid = pygame.font.SysFont(None, 22)
            self.font_ui = pygame.font.SysFont(None, 20)
            self.font_ui_bold = pygame.font.SysFont(None, 22)
            self.font_msg = pygame.font.SysFont(None, 36)

        # State variables initialized in reset()
        self.grid = None
        self.hidden_words = None
        self.cursor_pos = None
        self.selected_path = None
        self.found_words = None
        self.score = None
        self.steps = None
        self.timer = None
        self.game_over = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.feedback_msg = None
        self.feedback_timer = None
        self.feedback_color = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.TOTAL_TIME
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.selected_path = []
        self.found_words = set()
        self.prev_space_held = False
        self.prev_shift_held = False
        self.feedback_msg = ""
        self.feedback_timer = 0
        
        self._generate_grid()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # Update game state
        self._handle_input(movement, space_held, shift_held)
        
        # Handle word submission
        shift_pressed = shift_held and not self.prev_shift_held
        if shift_pressed and self.selected_path:
            reward += self._submit_word()
        
        # Update timers
        self.steps += 1
        self.timer -= 1.0 / 30.0 # Assuming 30 FPS
        if self.feedback_timer > 0:
            self.feedback_timer -= 1

        # Update previous action states
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if len(self.found_words) == len(self.WORD_LIST):
                reward += 100 # Win bonus
            else:
                reward -= 100 # Time out penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # --- Cursor Movement ---
        prev_cursor_pos = list(self.cursor_pos)
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)
        cursor_moved = prev_cursor_pos != self.cursor_pos

        # --- Selection Logic ---
        space_pressed = space_held and not self.prev_space_held
        
        if space_pressed:
            if not self.selected_path: # Start selection
                self.selected_path = [list(self.cursor_pos)]
            else: # Cancel selection
                self.selected_path = []
        elif not space_held and self.selected_path: # Cancel on release
            self.selected_path = []

        # "Drag" to select
        if cursor_moved and space_held and self.selected_path:
            last_pos = self.selected_path[-1]
            dx, dy = self.cursor_pos[0] - last_pos[0], self.cursor_pos[1] - last_pos[1]
            # Check adjacency (including diagonals) and uniqueness
            if max(abs(dx), abs(dy)) == 1 and list(self.cursor_pos) not in self.selected_path:
                self.selected_path.append(list(self.cursor_pos))

    def _submit_word(self):
        if not self.selected_path:
            return 0
        
        selected_word = "".join([self.grid[y][x] for x, y in self.selected_path])
        
        is_correct = False
        for hidden_word_info in self.hidden_words.values():
            if hidden_word_info["word"] == selected_word or hidden_word_info["word"] == selected_word[::-1]:
                is_correct = True
                word_to_check = hidden_word_info["word"]
                break
        
        if is_correct:
            if word_to_check not in self.found_words:
                self.found_words.add(word_to_check)
                self.score += len(word_to_check) * 10
                self._set_feedback("CORRECT!", self.COLOR_MSG_CORRECT, 45)
                # Sound: Correct word found
                return 10.0 # Reward for finding a new word
            else:
                self._set_feedback("ALREADY FOUND", self.COLOR_MSG_DUPE, 30)
                # Sound: Duplicate word
                return 0
        else:
            self._set_feedback("INCORRECT", self.COLOR_MSG_INCORRECT, 30)
            # Sound: Incorrect word
            return -1.0 # Penalty for incorrect submission

    def _check_termination(self):
        win = len(self.found_words) == len(self.WORD_LIST)
        timeout = self.timer <= 0
        max_steps_reached = self.steps >= self.MAX_STEPS
        return win or timeout or max_steps_reached

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "words_found": len(self.found_words),
            "words_total": len(self.WORD_LIST),
        }

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=8)

        # Draw highlights for found words (underneath letters)
        found_cells = set()
        for word, info in self.hidden_words.items():
            if word in self.found_words:
                for pos in info["path"]:
                    found_cells.add(tuple(pos))
        
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if (c, r) in found_cells:
                    cell_rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_FOUND_WORD, cell_rect)

        # Draw selection highlight
        if self.selected_path:
            for i, pos in enumerate(self.selected_path):
                c, r = pos
                cell_rect = pygame.Rect(self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                
                # Use a surface for transparency
                s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                s.fill(self.COLOR_SELECTION)
                self.screen.blit(s, cell_rect.topleft)

        # Draw grid lines and letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Lines
                if c > 0:
                    pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_X + c * self.CELL_SIZE, self.GRID_Y), (self.GRID_X + c * self.CELL_SIZE, self.GRID_Y + self.GRID_HEIGHT))
                if r > 0:
                    pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (self.GRID_X, self.GRID_Y + r * self.CELL_SIZE), (self.GRID_X + self.GRID_WIDTH, self.GRID_Y + r * self.CELL_SIZE))
                
                # Letters
                letter = self.grid[r][c]
                text_surf = self.font_grid.render(letter, True, self.COLOR_TEXT)
                text_rect = text_surf.get_rect(center=(self.GRID_X + c * self.CELL_SIZE + self.CELL_SIZE // 2, self.GRID_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2))
                self.screen.blit(text_surf, text_rect)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_X + cx * self.CELL_SIZE, self.GRID_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)

    def _render_ui(self):
        # Top UI Bar
        ui_bar_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, 40)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_bar_rect)
        pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (0, 40), (self.SCREEN_WIDTH, 40))

        # Score
        score_text = self.font_ui_bold.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 10))

        # Timer
        timer_color = self.COLOR_TEXT if self.timer > 10 else self.COLOR_TIMER_LOW
        timer_text = self.font_ui_bold.render(f"TIME: {max(0, math.ceil(self.timer))}", True, timer_color)
        timer_rect = timer_text.get_rect(centerx=self.SCREEN_WIDTH // 2)
        timer_rect.y = 10
        self.screen.blit(timer_text, timer_rect)

        # Word List
        word_list_x = self.GRID_X + self.GRID_WIDTH + 20
        word_list_y = self.GRID_Y - 10
        found_count = 0
        for i, word in enumerate(self.WORD_LIST):
            y_pos = word_list_y + i * 20
            if y_pos > self.SCREEN_HEIGHT - 20: continue # Don't draw off-screen
            
            if word in self.found_words:
                found_count += 1
                text_surf = self.font_ui.render(word, True, self.COLOR_FOUND_STRIKE)
                self.screen.blit(text_surf, (word_list_x, y_pos))
                pygame.draw.line(self.screen, self.COLOR_FOUND_STRIKE, (word_list_x, y_pos + 10), (word_list_x + text_surf.get_width(), y_pos + 10), 1)
            else:
                text_surf = self.font_ui.render(word, True, self.COLOR_TEXT)
                self.screen.blit(text_surf, (word_list_x, y_pos))
        
        # Found Count
        found_text = self.font_ui_bold.render(f"FOUND: {found_count}/{len(self.WORD_LIST)}", True, self.COLOR_TEXT)
        found_rect = found_text.get_rect(right=self.SCREEN_WIDTH - 15)
        found_rect.y = 10
        self.screen.blit(found_text, found_rect)

        # Feedback Message
        if self.feedback_timer > 0:
            alpha = min(255, int(255 * (self.feedback_timer / 20.0))) # Fade out
            msg_surf = self.font_msg.render(self.feedback_msg, True, self.feedback_color)
            msg_surf.set_alpha(alpha)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def _set_feedback(self, msg, color, duration):
        self.feedback_msg = msg
        self.feedback_color = color
        self.feedback_timer = duration

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.hidden_words = {}
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]
        
        for word in sorted(self.WORD_LIST, key=len, reverse=True):
            placed = False
            for _ in range(200): # Max attempts to place a word
                self.np_random.shuffle(directions)
                d = directions[0]
                
                start_c = self.np_random.integers(0, self.GRID_COLS)
                start_r = self.np_random.integers(0, self.GRID_ROWS)
                
                end_c = start_c + (len(word) - 1) * d[0]
                end_r = start_r + (len(word) - 1) * d[1]
                
                if not (0 <= end_c < self.GRID_COLS and 0 <= end_r < self.GRID_ROWS):
                    continue

                can_place = True
                path = []
                for i in range(len(word)):
                    c, r = start_c + i * d[0], start_r + i * d[1]
                    path.append((c,r))
                    if self.grid[r][c] != '' and self.grid[r][c] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    path_to_store = []
                    for i in range(len(word)):
                        c, r = start_c + i * d[0], start_r + i * d[1]
                        self.grid[r][c] = word[i]
                        path_to_store.append([c,r])
                    self.hidden_words[word] = {"word": word, "path": path_to_store}
                    placed = True
                    break
            
            if not placed:
                # This should be rare with a large enough grid
                # For robustness, we could retry the whole grid generation
                # print(f"Warning: Could not place word '{word}'. Retrying grid generation.")
                return self._generate_grid()

        # Fill empty spaces with random letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == '':
                    self.grid[r][c] = self.np_random.choice(list(alphabet))

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for manual play
    render_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption(GameEnv.game_description)
    clock = pygame.time.Clock()
    
    # Game loop
    running = True
    while running:
        # --- Action mapping for human player ---
        movement = 0 # no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = keys[pygame.K_SPACE]
        shift_held = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        
        action = [movement, 1 if space_held else 0, 1 if shift_held else 0]
        
        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']}, Terminated: {terminated}")

        # --- Rendering ---
        # The observation is already the rendered screen, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        render_screen.blit(surf, (0, 0))
        pygame.display.flip()

        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("--- RESETTING ---")
                obs, info = env.reset()

        if terminated or truncated:
            print("--- GAME OVER ---")
            print(f"Final Score: {info['score']}")
            time.sleep(2) # Pause before resetting
            obs, info = env.reset()

        clock.tick(30) # Run at 30 FPS

    env.close()