
# Generated: 2025-08-27T21:44:26.682509
# Source Brief: brief_02892.md
# Brief Index: 2892

        
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
        "Controls: ↑↓←→ to move the cursor. Press space to select the first and last letter of a word. Hold shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find hidden words in a grid against the clock. Select the start and end letters to claim a word."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 14
        self.NUM_WORDS = 10
        self.MAX_TIME = 60  # seconds
        self.FPS = 30
        self.MAX_STEPS = self.MAX_TIME * self.FPS

        self.WORD_LIST = [
            "PYTHON", "GYMNASIUM", "AGENT", "REWARD", "ACTION", "STATE",
            "POLICY", "LEARNING", "NEURAL", "NETWORK", "VECTOR", "TENSOR",
            "EPISODE", "TRAIN", "MODEL", "SOLVE", "SEARCH", "GRID", "PUZZLE"
        ]

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID = (40, 50, 60)
        self.COLOR_TEXT = (200, 210, 220)
        self.COLOR_CURSOR = (255, 180, 0, 100)
        self.COLOR_SELECT = (0, 150, 255, 150)
        self.COLOR_CORRECT = (0, 255, 100)
        self.COLOR_INCORRECT = (255, 50, 50)
        self.COLOR_UI_BG = (30, 40, 50)
        self.COLOR_UI_HEADER = (255, 180, 0)
        self.COLOR_FOUND_WORD = (100, 120, 140)

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font_grid = pygame.font.SysFont("Consolas", 18, bold=True)
            self.font_ui_header = pygame.font.SysFont("Arial", 20, bold=True)
            self.font_ui_text = pygame.font.SysFont("Arial", 16)
            self.font_timer = pygame.font.SysFont("Consolas", 32, bold=True)
        except pygame.error:
            self.font_grid = pygame.font.Font(None, 24)
            self.font_ui_header = pygame.font.Font(None, 28)
            self.font_ui_text = pygame.font.Font(None, 22)
            self.font_timer = pygame.font.Font(None, 40)
        
        # Grid layout
        self.GRID_AREA_WIDTH = self.HEIGHT
        self.CELL_SIZE = self.GRID_AREA_WIDTH // self.GRID_SIZE
        self.GRID_OFFSET_X = (self.GRID_AREA_WIDTH - self.CELL_SIZE * self.GRID_SIZE) // 2
        self.GRID_OFFSET_Y = (self.HEIGHT - self.CELL_SIZE * self.GRID_SIZE) // 2
        
        # Initialize state variables
        self.grid = []
        self.words_to_find = []
        self.word_data = {}
        self.cursor_pos = [0, 0]
        self.selection_start = None
        self.last_space_held = False
        self.feedback_anims = []
        self.steps = 0
        self.score = 0
        self.time_remaining = 0
        self.game_over = False

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_TIME
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selection_start = None
        self.last_space_held = False
        self.feedback_anims = []
        
        self._generate_grid()
        
        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.words_to_find = self.np_random.choice(self.WORD_LIST, self.NUM_WORDS, replace=False).tolist()
        self.word_data = {}

        for word in self.words_to_find:
            placed = False
            for _ in range(100): # 100 attempts to place a word
                direction = self.np_random.choice([(0, 1), (1, 0), (1, 1), (1, -1)])
                if self.np_random.random() < 0.5:
                    direction = (-direction[0], -direction[1])
                
                start_x = self.np_random.integers(0, self.GRID_SIZE)
                start_y = self.np_random.integers(0, self.GRID_SIZE)
                
                end_x = start_x + (len(word) - 1) * direction[0]
                end_y = start_y + (len(word) - 1) * direction[1]

                if not (0 <= end_x < self.GRID_SIZE and 0 <= end_y < self.GRID_SIZE):
                    continue

                can_place = True
                all_coords = []
                for i in range(len(word)):
                    x = start_x + i * direction[0]
                    y = start_y + i * direction[1]
                    all_coords.append((x, y))
                    if self.grid[y][x] != '' and self.grid[y][x] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i in range(len(word)):
                        x = start_x + i * direction[0]
                        y = start_y + i * direction[1]
                        self.grid[y][x] = word[i]
                    
                    self.word_data[word] = {
                        "start": (start_x, start_y),
                        "end": (end_x, end_y),
                        "all_coords": set(all_coords),
                        "found": False
                    }
                    placed = True
                    break
        
        # Fill empty spaces with random letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == '':
                    self.grid[y][x] = self.np_random.choice(list(alphabet))

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        space_press = space_held and not self.last_space_held
        self.last_space_held = space_held

        reward = self._handle_input(movement, space_press, shift_held)
        self._update_game_state()
        
        self.steps += 1
        terminated = self._check_termination()

        if terminated and sum(1 for d in self.word_data.values() if d["found"]) == self.NUM_WORDS:
            reward += 50
        
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, movement, space_press, shift_held):
        reward = 0
        
        # 1. Handle deselection
        if shift_held and self.selection_start:
            self.selection_start = None
            # sound: deselect
        
        # 2. Handle movement
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # 3. Handle selection
        if space_press:
            # sound: select
            cursor_tuple = tuple(self.cursor_pos)
            if self.selection_start is None:
                self.selection_start = cursor_tuple
                
                # Continuous reward for selecting a potentially correct letter
                is_part_of_word = False
                for data in self.word_data.values():
                    if not data["found"] and cursor_tuple in data["all_coords"]:
                        is_part_of_word = True
                        break
                reward += 0.1 if is_part_of_word else -0.1
            else:
                word_found = self._check_selection(self.selection_start, cursor_tuple)
                if word_found:
                    reward += 10
                    self.word_data[word_found]["found"] = True
                    start = self.word_data[word_found]["start"]
                    end = self.word_data[word_found]["end"]
                    self.feedback_anims.append({"type": "correct", "start": start, "end": end, "timer": self.FPS // 2})
                    # sound: correct_word
                else:
                    self.feedback_anims.append({"type": "incorrect", "start": self.selection_start, "end": cursor_tuple, "timer": self.FPS // 3})
                    # sound: incorrect_word
                self.selection_start = None

        return reward
    
    def _check_selection(self, start_pos, end_pos):
        for word, data in self.word_data.items():
            if not data["found"]:
                if (data["start"] == start_pos and data["end"] == end_pos) or \
                   (data["start"] == end_pos and data["end"] == start_pos):
                    return word
        return None

    def _update_game_state(self):
        self.time_remaining -= 1 / self.FPS
        
        # Update animations
        self.feedback_anims = [anim for anim in self.feedback_anims if anim["timer"] > 0]
        for anim in self.feedback_anims:
            anim["timer"] -= 1

    def _check_termination(self):
        if self.time_remaining <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        if all(d["found"] for d in self.word_data.values()):
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y)
            end_pos = (self.GRID_OFFSET_X + i * self.CELL_SIZE, self.GRID_OFFSET_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_OFFSET_X, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_OFFSET_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_OFFSET_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

        # Draw letters
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                letter = self.grid[y][x]
                text_surf = self.font_grid.render(letter, True, self.COLOR_TEXT)
                text_rect = text_surf.get_rect(center=(
                    self.GRID_OFFSET_X + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.GRID_OFFSET_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2
                ))
                self.screen.blit(text_surf, text_rect)

        # Draw selection highlight
        if self.selection_start:
            sx, sy = self.selection_start
            rect = pygame.Rect(
                self.GRID_OFFSET_X + sx * self.CELL_SIZE,
                self.GRID_OFFSET_Y + sy * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_SELECT)
            self.screen.blit(s, rect.topleft)

        # Draw cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_OFFSET_X + cx * self.CELL_SIZE,
            self.GRID_OFFSET_Y + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_CURSOR)
        self.screen.blit(s, rect.topleft)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR[:3], rect, 2)
        
        # Draw feedback animations
        for anim in self.feedback_anims:
            color = self.COLOR_CORRECT if anim["type"] == "correct" else self.COLOR_INCORRECT
            start_pixel = (
                self.GRID_OFFSET_X + anim["start"][0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                self.GRID_OFFSET_Y + anim["start"][1] * self.CELL_SIZE + self.CELL_SIZE // 2
            )
            end_pixel = (
                self.GRID_OFFSET_X + anim["end"][0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                self.GRID_OFFSET_Y + anim["end"][1] * self.CELL_SIZE + self.CELL_SIZE // 2
            )
            pygame.draw.line(self.screen, color, start_pixel, end_pixel, 4)
            pygame.gfxdraw.filled_circle(self.screen, start_pixel[0], start_pixel[1], 8, color)
            pygame.gfxdraw.filled_circle(self.screen, end_pixel[0], end_pixel[1], 8, color)
            
    def _render_ui(self):
        ui_x = self.GRID_AREA_WIDTH + 10
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, (ui_x - 10, 0, self.WIDTH - ui_x + 10, self.HEIGHT))
        
        # Timer
        timer_text = f"{max(0, self.time_remaining):.1f}"
        timer_surf = self.font_timer.render(timer_text, True, self.COLOR_UI_HEADER)
        timer_rect = timer_surf.get_rect(topright=(self.WIDTH - 20, 10))
        self.screen.blit(timer_surf, timer_rect)

        # Score
        score_text = f"Score: {int(self.score)}"
        score_surf = self.font_ui_header.render(score_text, True, self.COLOR_TEXT)
        score_rect = score_surf.get_rect(topleft=(ui_x, 15))
        self.screen.blit(score_surf, score_rect)
        
        # Word list
        word_list_y = 70
        header_surf = self.font_ui_header.render("Words to Find:", True, self.COLOR_UI_HEADER)
        self.screen.blit(header_surf, (ui_x, word_list_y))
        word_list_y += 30

        for word in self.words_to_find:
            is_found = self.word_data[word]["found"]
            color = self.COLOR_FOUND_WORD if is_found else self.COLOR_TEXT
            word_surf = self.font_ui_text.render(word, True, color)
            self.screen.blit(word_surf, (ui_x, word_list_y))
            if is_found:
                line_y = word_list_y + word_surf.get_height() // 2
                pygame.draw.line(self.screen, self.COLOR_INCORRECT, (ui_x, line_y), (ui_x + word_surf.get_width(), line_y), 2)
            word_list_y += 22

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_found": sum(1 for d in self.word_data.values() if d["found"]),
            "time_remaining": self.time_remaining
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
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Word Search")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Action defaults
        movement = 0 # None
        space = 0 # Released
        shift = 0 # Released

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
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Words Found: {info['words_found']}/{env.NUM_WORDS}")
            obs, info = env.reset()

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)
        
    env.close()