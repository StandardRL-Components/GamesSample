
# Generated: 2025-08-28T02:12:56.574899
# Source Brief: brief_01635.md
# Brief Index: 1635

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import OrderedDict
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move the cursor and create a selection path. "
        "Press space to submit the selected word. Press shift to clear your selection."
    )

    game_description = (
        "Find hidden words in a procedurally generated grid within a time limit. "
        "Each stage increases the difficulty."
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_SIZE = 12
    CELL_SIZE = 30
    GRID_WIDTH = GRID_HEIGHT = GRID_SIZE * CELL_SIZE
    GRID_ORIGIN_X = (SCREEN_WIDTH - GRID_WIDTH) // 4
    GRID_ORIGIN_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    FPS = 30
    TIME_PER_STAGE = 60  # seconds

    # --- Colors ---
    COLOR_BG = (28, 33, 48)
    COLOR_GRID_LINES = (48, 53, 68)
    COLOR_TEXT = (220, 220, 230)
    COLOR_CURSOR = (255, 200, 0, 150)
    COLOR_SELECTION = (255, 180, 0, 80)
    COLOR_UI_TEXT = (200, 200, 210)
    COLOR_TIMER = (255, 230, 180)
    COLOR_SCORE = (180, 255, 230)
    COLOR_FEEDBACK_CORRECT = (0, 255, 128)
    COLOR_FEEDBACK_INCORRECT = (255, 80, 80)
    COLOR_FOUND_WORD_BG = (60, 140, 100, 100)
    COLOR_FOUND_WORD_TEXT = (150, 255, 200)

    # --- Word Lists per Stage ---
    WORD_BANK = {
        1: ["CAT", "DOG", "SUN", "SKY", "RUN", "FLY", "SEE", "ONE", "TWO", "TEN"],
        2: ["WORD", "GAME", "CODE", "GRID", "FIND", "TIME", "PLAY", "GYM", "LOOP", "STEP"],
        3: ["PYTHON", "SEARCH", "PUZZLE", "VISUAL", "ACTION", "REWARD", "AGENT", "STATE"],
        4: ["EXPERT", "MINIMAL", "POLISHED", "ARCADE", "FLUID", "EFFECT", "DYNAMIC"],
    }
    MAX_STAGES = len(WORD_BANK)


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
        
        self.font_grid = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_ui_title = pygame.font.SysFont("Verdana", 18, bold=True)
        self.font_ui_body = pygame.font.SysFont("Verdana", 14)
        self.font_feedback = pygame.font.SysFont("Verdana", 16, bold=True)
        
        self.grid = []
        self.word_solutions = {}
        self.words_to_find = []
        self.found_words = set()
        self.found_word_coords = set()
        
        self.cursor_pos = [0, 0]
        self.selection_path = []
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.time_remaining = 0
        self.game_over = False
        
        self.space_pressed_last_frame = False
        self.shift_pressed_last_frame = False
        
        self.feedback_messages = []
        self.particles = []

        self.np_random = None

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.stage = 1
        self.game_over = False
        
        self._generate_new_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = self.game_over

        if not terminated:
            movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
            
            # --- Time and Termination Check ---
            self.time_remaining -= 1
            if self.time_remaining <= 0:
                self.game_over = True
                terminated = True
                self.feedback_messages.append(self._create_feedback("TIME UP!", self.COLOR_FEEDBACK_INCORRECT, 90))

            # --- Handle Actions ---
            reward += self._handle_input(movement, space_held, shift_held)

            self.steps += 1
        
        # --- Update Game State (Particles, Feedback) ---
        self._update_particles()
        self._update_feedback()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _generate_new_stage(self):
        self.time_remaining = self.TIME_PER_STAGE * self.FPS
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selection_path = []
        self.found_words = set()
        self.found_word_coords = set()
        self.feedback_messages = []
        
        self.words_to_find = self.WORD_BANK.get(self.stage, [])
        self._generate_grid()

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.word_solutions = {}
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        
        for word in self.words_to_find:
            placed = False
            attempts = 0
            while not placed and attempts < 100:
                attempts += 1
                self.np_random.shuffle(directions)
                d = directions[0]
                
                start_x = self.np_random.integers(0, self.GRID_SIZE)
                start_y = self.np_random.integers(0, self.GRID_SIZE)
                
                end_x = start_x + (len(word) - 1) * d[0]
                end_y = start_y + (len(word) - 1) * d[1]
                
                if 0 <= end_x < self.GRID_SIZE and 0 <= end_y < self.GRID_SIZE:
                    can_place = True
                    coords = []
                    for i in range(len(word)):
                        x, y = start_x + i * d[0], start_y + i * d[1]
                        coords.append((x, y))
                        if self.grid[y][x] != '' and self.grid[y][x] != word[i]:
                            can_place = False
                            break
                    
                    if can_place:
                        for i, (x, y) in enumerate(coords):
                            self.grid[y][x] = word[i]
                        self.word_solutions[word] = tuple(sorted(coords))
                        self.word_solutions[word[::-1]] = tuple(sorted(coords))
                        placed = True

        # Fill empty cells
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                if self.grid[y][x] == '':
                    self.grid[y][x] = self.np_random.choice(list(alphabet))

    def _handle_input(self, movement, space_held, shift_held):
        reward = 0
        
        # --- Movement and Selection Path ---
        if movement > 0:
            if not self.selection_path:
                self.selection_path.append(tuple(self.cursor_pos))

            if movement == 1: self.cursor_pos[1] -= 1  # Up
            elif movement == 2: self.cursor_pos[1] += 1  # Down
            elif movement == 3: self.cursor_pos[0] -= 1  # Left
            elif movement == 4: self.cursor_pos[0] += 1  # Right
            
            self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)
            
            if tuple(self.cursor_pos) not in self.selection_path:
                 self.selection_path.append(tuple(self.cursor_pos))

        # --- Shift (Clear Selection) ---
        if shift_held and not self.shift_pressed_last_frame:
            if self.selection_path:
                self.selection_path = []
                reward -= 0.1 # Penalty for cancelling
        
        # --- Space (Submit Word) ---
        if space_held and not self.space_pressed_last_frame:
            if len(self.selection_path) > 1:
                selected_word = "".join([self.grid[y][x] for x, y in self.selection_path])
                
                # Check if word is valid and not already found
                if selected_word in self.word_solutions and selected_word not in self.found_words and selected_word[::-1] not in self.found_words:
                    word_to_add = selected_word if selected_word in self.words_to_find else selected_word[::-1]
                    self.found_words.add(word_to_add)
                    
                    coords = self.word_solutions[selected_word]
                    for coord in coords:
                        self.found_word_coords.add(coord)

                    reward += 10 # Found a word
                    self.score += 100
                    self.feedback_messages.append(self._create_feedback(f"+100 '{word_to_add}'", self.COLOR_FEEDBACK_CORRECT, 60))
                    self._spawn_word_particles(self.selection_path)
                    
                    # Check for stage/game completion
                    if len(self.found_words) == len(self.words_to_find):
                        reward += 50 # Stage complete
                        self.score += 500
                        self.stage += 1
                        if self.stage > self.MAX_STAGES:
                            self.game_over = True
                            terminated = True
                            self.feedback_messages.append(self._create_feedback("ALL STAGES CLEAR!", self.COLOR_FEEDBACK_CORRECT, 120))
                        else:
                            self.feedback_messages.append(self._create_feedback("STAGE CLEAR!", self.COLOR_FEEDBACK_CORRECT, 90))
                            self._generate_new_stage()
                else:
                    reward -= 1 # Incorrect submission
                    self.score -= 10
                    self.feedback_messages.append(self._create_feedback("INCORRECT", self.COLOR_FEEDBACK_INCORRECT, 45))

            self.selection_path = []
        
        self.space_pressed_last_frame = space_held
        self.shift_pressed_last_frame = shift_held
        
        return reward

    # --- Rendering ---

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            x = self.GRID_ORIGIN_X + i * self.CELL_SIZE
            y = self.GRID_ORIGIN_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_ORIGIN_Y), (x, self.GRID_ORIGIN_Y + self.GRID_HEIGHT))
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_ORIGIN_X, y), (self.GRID_ORIGIN_X + self.GRID_WIDTH, y))

        # Draw found word backgrounds
        for x, y in self.found_word_coords:
            rect = pygame.Rect(self.GRID_ORIGIN_X + x * self.CELL_SIZE, self.GRID_ORIGIN_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_FOUND_WORD_BG)
            self.screen.blit(s, rect.topleft)

        # Draw selection path
        if self.selection_path:
            for x, y in self.selection_path:
                rect = pygame.Rect(self.GRID_ORIGIN_X + x * self.CELL_SIZE, self.GRID_ORIGIN_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                s.fill(self.COLOR_SELECTION)
                self.screen.blit(s, rect.topleft)

        # Draw letters
        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                letter = self.grid[y][x]
                color = self.COLOR_FOUND_WORD_TEXT if (x, y) in self.found_word_coords else self.COLOR_TEXT
                text_surf = self.font_grid.render(letter, True, color)
                text_rect = text_surf.get_rect(center=(self.GRID_ORIGIN_X + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                                                       self.GRID_ORIGIN_Y + y * self.CELL_SIZE + self.CELL_SIZE // 2))
                self.screen.blit(text_surf, text_rect)

        # Draw cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.GRID_ORIGIN_X + cx * self.CELL_SIZE, self.GRID_ORIGIN_Y + cy * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, self.COLOR_CURSOR, s.get_rect(), border_radius=4)
        self.screen.blit(s, cursor_rect.topleft)

        # Draw particles
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = p['color'][:3] + (alpha,)
            s = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            pygame.draw.circle(s, color, (p['size'] // 2, p['size'] // 2), p['size'] // 2)
            self.screen.blit(s, (int(p['pos'][0]), int(p['pos'][1])))

        # Draw feedback messages
        for msg in self.feedback_messages:
            text_surf = self.font_feedback.render(msg['text'], True, msg['color'])
            text_rect = text_surf.get_rect(center=msg['pos'])
            self.screen.blit(text_surf, text_rect)

    def _render_ui(self):
        ui_x = self.GRID_ORIGIN_X + self.GRID_WIDTH + 40
        
        # --- Timer ---
        time_text = f"{self.time_remaining / self.FPS:.1f}"
        self._draw_text("TIME", self.font_ui_title, self.COLOR_TIMER, ui_x, 30)
        self._draw_text(time_text, self.font_ui_body, self.COLOR_UI_TEXT, ui_x, 55)

        # --- Score ---
        self._draw_text("SCORE", self.font_ui_title, self.COLOR_SCORE, ui_x, 95)
        self._draw_text(str(self.score), self.font_ui_body, self.COLOR_UI_TEXT, ui_x, 120)

        # --- Word List ---
        self._draw_text(f"WORDS (STAGE {self.stage})", self.font_ui_title, self.COLOR_UI_TEXT, ui_x, 160)
        y_offset = 185
        for word in self.words_to_find:
            if word in self.found_words:
                surf = self.font_ui_body.render(word, True, self.COLOR_FOUND_WORD_TEXT)
                pygame.draw.line(surf, self.COLOR_FOUND_WORD_TEXT, (0, surf.get_height()//2), (surf.get_width(), surf.get_height()//2), 2)
            else:
                surf = self.font_ui_body.render(word, True, self.COLOR_UI_TEXT)
            self.screen.blit(surf, (ui_x, y_offset))
            y_offset += 20

    def _draw_text(self, text, font, color, x, y, center=False):
        text_surf = font.render(text, True, color)
        if center:
            text_rect = text_surf.get_rect(center=(x, y))
        else:
            text_rect = text_surf.get_rect(topleft=(x, y))
        self.screen.blit(text_surf, text_rect)

    # --- Helpers & Effects ---

    def _create_feedback(self, text, color, duration):
        return {
            "text": text,
            "color": color,
            "duration": duration,
            "pos": (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 30)
        }

    def _update_feedback(self):
        for msg in self.feedback_messages[:]:
            msg['duration'] -= 1
            if msg['duration'] <= 0:
                self.feedback_messages.remove(msg)

    def _spawn_word_particles(self, path):
        for x, y in path:
            for _ in range(5):
                pos = [self.GRID_ORIGIN_X + x * self.CELL_SIZE + self.CELL_SIZE / 2,
                       self.GRID_ORIGIN_Y + y * self.CELL_SIZE + self.CELL_SIZE / 2]
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                life = self.np_random.integers(15, 30)
                self.particles.append({
                    'pos': pos, 'vel': vel, 'life': life, 'max_life': life,
                    'color': self.COLOR_FEEDBACK_CORRECT, 'size': self.np_random.integers(3, 7)
                })

    def _update_particles(self):
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.stage,
            "words_found": len(self.found_words),
            "words_total": len(self.words_to_find),
        }

    def close(self):
        pygame.quit()

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

if __name__ == '__main__':
    # To play the game manually, you can run this file.
    # This is for demonstration and debugging purposes.
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Search Gym Environment")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0
        
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
        total_reward += reward
        
        # Draw the observation from the environment to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            pygame.time.wait(3000)
            obs, info = env.reset()
            total_reward = 0
            
        clock.tick(GameEnv.FPS)
        
    env.close()