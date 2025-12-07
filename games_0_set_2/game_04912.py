
# Generated: 2025-08-28T03:23:30.372391
# Source Brief: brief_04912.md
# Brief Index: 4912

        
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


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.x = x
        self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.lifetime = random.randint(20, 40)
        self.size = random.uniform(2, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifetime > 0:
            pos = (int(self.x), int(self.y))
            pygame.gfxdraw.filled_circle(surface, pos[0], pos[1], int(self.size), self.color)
            pygame.gfxdraw.aacircle(surface, pos[0], pos[1], int(self.size), self.color)

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to start selecting a word, and Shift to submit it."
    )

    game_description = (
        "Find all the hidden words in the grid before the time runs out!"
    )

    auto_advance = True

    # --- Constants ---
    GRID_SIZE = 20
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_AREA_WIDTH = 400
    UI_AREA_WIDTH = SCREEN_WIDTH - GRID_AREA_WIDTH
    CELL_SIZE = GRID_AREA_WIDTH // GRID_SIZE
    MAX_TIME = 60  # seconds

    # --- Colors ---
    COLOR_BG = (25, 28, 36)
    COLOR_GRID_LINES = (50, 55, 65)
    COLOR_LETTER = (200, 205, 215)
    COLOR_CURSOR = (70, 140, 255)
    COLOR_SELECTION = (255, 220, 100, 100) # RGBA for transparency
    COLOR_FOUND_WORD = (80, 200, 120, 120) # RGBA
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEADER = (100, 180, 255)
    COLOR_UI_FOUND = (120, 120, 120)
    COLOR_CORRECT = (80, 255, 120)
    COLOR_INCORRECT = (255, 80, 80)
    
    WORD_LIST = ["PYTHON", "GYMNASIUM", "AGENT", "REWARD", "ACTION", "STATE", "POLICY", "PUZZLE", "SEARCH", "VECTOR", "VISUAL", "EXPERT"]


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
        self.fps = 30
        
        self.font_letter = pygame.font.Font(None, 22)
        self.font_ui_header = pygame.font.Font(None, 28)
        self.font_ui_text = pygame.font.Font(None, 24)
        self.font_feedback = pygame.font.Font(None, 60)
        self.font_word_list = pygame.font.Font(None, 20)

        self.selection_surface = pygame.Surface((self.GRID_AREA_WIDTH, self.GRID_AREA_WIDTH), pygame.SRCALPHA)

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME * self.fps

        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.is_selecting = False
        self.selection_start_pos = None
        
        self.found_words_data = {} # word -> list of (x,y) coords
        self.word_locations = {} # word -> (start_pos, end_pos)

        while not self._generate_grid():
            pass # Keep trying until a valid grid is generated

        self.particles = []
        self.feedback_message = None
        self.feedback_timer = 0
        self.feedback_color = self.COLOR_CORRECT
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        self.time_remaining = max(0, self.time_remaining - 1)
        
        movement, space_pressed, shift_pressed = action[0], action[1] == 1, action[2] == 1
        
        self._handle_input(movement, space_pressed, shift_pressed)
        
        if shift_pressed and self.is_selecting:
            reward += self._process_submission()

        self._update_particles()
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        else:
            self.feedback_message = None

        self.score += reward
        terminated = self._check_termination()

        if terminated and not self.game_over:
            self.game_over = True
            if len(self.found_words_data) == len(self.WORD_LIST):
                final_reward = 100
                reward += final_reward
                self.score += final_reward
                self._show_feedback("YOU WIN!", self.COLOR_CORRECT, 90)
            else:
                self._show_feedback("TIME'S UP!", self.COLOR_INCORRECT, 90)

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, movement, space_pressed, shift_pressed):
        if not self.is_selecting:
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        
        if space_pressed and not self.is_selecting:
            self.is_selecting = True
            self.selection_start_pos = tuple(self.cursor_pos)
        
        if self.is_selecting:
            # Allow cursor movement while selecting
            if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
            elif movement == 2: self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)
            elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
            elif movement == 4: self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)


    def _process_submission(self):
        reward = 0
        word, path = self._get_selected_word_and_path()
        
        is_correct = word is not None and word in self.WORD_LIST and word not in self.found_words_data
        is_correct_reversed = False
        if not is_correct and word is not None:
             reversed_word = word[::-1]
             is_correct_reversed = reversed_word in self.WORD_LIST and reversed_word not in self.found_words_data
             if is_correct_reversed:
                 word = reversed_word
                 path = path[::-1]

        if is_correct or is_correct_reversed:
            reward = 10
            self.found_words_data[word] = path
            self._show_feedback("CORRECT!", self.COLOR_CORRECT, 30)
            # Find center of word for particle explosion
            center_x = (path[0][0] + path[-1][0]) / 2 * self.CELL_SIZE + self.CELL_SIZE / 2
            center_y = (path[0][1] + path[-1][1]) / 2 * self.CELL_SIZE + self.CELL_SIZE / 2
            self._spawn_particles(center_x, center_y, 50, self.COLOR_CORRECT)
            # sfx: positive chime
        else:
            reward = -1
            self._show_feedback("INCORRECT", self.COLOR_INCORRECT, 30)
            # sfx: negative buzz

        self.is_selecting = False
        self.selection_start_pos = None
        return reward

    def _get_selected_word_and_path(self):
        if not self.selection_start_pos:
            return None, None

        start_x, start_y = self.selection_start_pos
        end_x, end_y = self.cursor_pos
        
        dx, dy = end_x - start_x, end_y - start_y
        
        path = []
        word = ""

        if dx == 0 and dy == 0: # Single letter
            path.append((start_x, start_y))
        elif dx == 0: # Vertical
            step = 1 if end_y > start_y else -1
            for y in range(start_y, end_y + step, step):
                path.append((start_x, y))
        elif dy == 0: # Horizontal
            step = 1 if end_x > start_x else -1
            for x in range(start_x, end_x + step, step):
                path.append((x, start_y))
        elif abs(dx) == abs(dy): # Diagonal
            step_x = 1 if end_x > start_x else -1
            step_y = 1 if end_y > start_y else -1
            for i in range(abs(dx) + 1):
                path.append((start_x + i * step_x, start_y + i * step_y))
        else: # Invalid line
            return None, None

        for x, y in path:
            word += self.grid[y][x]
        
        return word, path

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.word_locations.clear()
        
        temp_words = self.WORD_LIST[:]
        random.shuffle(temp_words)

        for word in temp_words:
            placed = False
            for _ in range(100): # 100 placement attempts per word
                direction = self.np_random.choice([
                    (1, 0), (0, 1), (1, 1), # H, V, D
                    (-1, 0), (0, -1), (-1, -1), # Reversed H, V, D
                    (1, -1), (-1, 1) # Other Diagonals
                ])
                
                start_x = self.np_random.integers(0, self.GRID_SIZE)
                start_y = self.np_random.integers(0, self.GRID_SIZE)
                
                end_x = start_x + (len(word) - 1) * direction[0]
                end_y = start_y + (len(word) - 1) * direction[1]

                if 0 <= end_x < self.GRID_SIZE and 0 <= end_y < self.GRID_SIZE:
                    can_place = True
                    for i in range(len(word)):
                        px, py = start_x + i * direction[0], start_y + i * direction[1]
                        if self.grid[py][px] != '' and self.grid[py][px] != word[i]:
                            can_place = False
                            break
                    
                    if can_place:
                        path = []
                        for i in range(len(word)):
                            px, py = start_x + i * direction[0], start_y + i * direction[1]
                            self.grid[py][px] = word[i]
                            path.append((px, py))
                        self.word_locations[word] = path
                        placed = True
                        break
            if not placed:
                return False # Failed to place a word, restart generation

        # Fill empty cells
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r][c] == '':
                    self.grid[r][c] = self.np_random.choice(list(string.ascii_uppercase))
        return True

    def _check_termination(self):
        return self.time_remaining <= 0 or len(self.found_words_data) == len(self.WORD_LIST)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self.selection_surface.fill((0,0,0,0))
        
        # Draw found word highlights
        for word, path in self.found_words_data.items():
            for x, y in path:
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.selection_surface, self.COLOR_FOUND_WORD, rect)

        # Draw current selection highlight
        if self.is_selecting:
            word, path = self._get_selected_word_and_path()
            if path:
                for x, y in path:
                    rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                    pygame.draw.rect(self.selection_surface, self.COLOR_SELECTION, rect)
        
        self.screen.blit(self.selection_surface, (0, 0))

        # Draw grid lines and letters
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                rect = pygame.Rect(x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, rect, 1)
                
                letter_surf = self.font_letter.render(self.grid[y][x], True, self.COLOR_LETTER)
                letter_rect = letter_surf.get_rect(center=rect.center)
                self.screen.blit(letter_surf, letter_rect)
        
        # Draw cursor
        cursor_rect = pygame.Rect(self.cursor_pos[0] * self.CELL_SIZE, self.cursor_pos[1] * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 2)

        self._render_particles()
        self._render_feedback()

    def _render_ui(self):
        # Top bar
        top_bar_y = 5
        time_text = f"TIME: {self.time_remaining / self.fps:.1f}"
        score_text = f"SCORE: {self.score}"
        found_text = f"FOUND: {len(self.found_words_data)}/{len(self.WORD_LIST)}"
        
        time_surf = self.font_ui_text.render(time_text, True, self.COLOR_UI_TEXT)
        score_surf = self.font_ui_text.render(score_text, True, self.COLOR_UI_TEXT)
        found_surf = self.font_ui_text.render(found_text, True, self.COLOR_UI_TEXT)
        
        self.screen.blit(time_surf, (self.GRID_AREA_WIDTH + 15, top_bar_y))
        self.screen.blit(score_surf, (self.GRID_AREA_WIDTH + 15, top_bar_y + 25))
        self.screen.blit(found_surf, (self.GRID_AREA_WIDTH + 15, top_bar_y + 50))
        
        # Word list
        word_list_y_start = top_bar_y + 90
        header_surf = self.font_ui_header.render("WORDS TO FIND", True, self.COLOR_UI_HEADER)
        self.screen.blit(header_surf, (self.GRID_AREA_WIDTH + 15, word_list_y_start))
        
        for i, word in enumerate(self.WORD_LIST):
            y_pos = word_list_y_start + 30 + i * 22
            is_found = word in self.found_words_data
            color = self.COLOR_UI_FOUND if is_found else self.COLOR_UI_TEXT
            
            word_surf = self.font_word_list.render(word, True, color)
            self.screen.blit(word_surf, (self.GRID_AREA_WIDTH + 25, y_pos))
            
            if is_found:
                pygame.draw.line(self.screen, self.COLOR_UI_FOUND, 
                                 (self.GRID_AREA_WIDTH + 25, y_pos + 10), 
                                 (self.GRID_AREA_WIDTH + 25 + word_surf.get_width(), y_pos + 10), 2)

    def _render_feedback(self):
        if self.feedback_message and self.feedback_timer > 0:
            feedback_surf = self.font_feedback.render(self.feedback_message, True, self.feedback_color)
            feedback_rect = feedback_surf.get_rect(center=(self.GRID_AREA_WIDTH / 2, self.GRID_AREA_WIDTH / 2))
            
            alpha = min(255, int(255 * (self.feedback_timer / 20)))
            feedback_surf.set_alpha(alpha)
            
            self.screen.blit(feedback_surf, feedback_rect)

    def _spawn_particles(self, x, y, count, color):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

    def _update_particles(self):
        for p in self.particles:
            p.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]

    def _render_particles(self):
        for p in self.particles:
            p.draw(self.screen)

    def _show_feedback(self, text, color, duration):
        self.feedback_message = text
        self.feedback_color = color
        self.feedback_timer = duration

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": self.time_remaining / self.fps,
            "words_found": len(self.found_words_data),
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Use a real screen for human play
    real_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Search Puzzle")
    
    terminated = False
    
    while not terminated:
        movement = 0 # no-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Draw the observation from the environment to the real screen
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        real_screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        env.clock.tick(env.fps)

    env.close()