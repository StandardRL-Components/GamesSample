
# Generated: 2025-08-28T05:45:44.283441
# Source Brief: brief_02721.md
# Brief Index: 2721

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set a dummy video driver to run Pygame headlessly
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press space to select letters and shift to submit the word."
    )

    game_description = (
        "Find hidden words in a procedurally generated grid before time runs out. "
        "Select letters in a continuous line to form words."
    )

    auto_advance = True
    
    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 14, 10
    GRID_X_OFFSET, GRID_Y_OFFSET = 40, 80
    CELL_SIZE = 38
    
    FPS = 30
    MAX_STEPS = 1800 # 60 seconds * 30 FPS
    
    # --- Colors ---
    COLOR_BG = (25, 35, 55)
    COLOR_GRID_LINES = (45, 55, 75)
    COLOR_LETTER = (200, 210, 230)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION = (0, 150, 255)
    COLOR_FOUND_WORD_BG = (40, 80, 60)
    COLOR_FOUND_WORD_FG = (100, 220, 150)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_VALUE = (255, 255, 255)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_FEEDBACK_GOOD = (0, 255, 100, 100)
    COLOR_FEEDBACK_BAD = (255, 0, 0, 100)
    
    WORD_LIST = [
        "PYTHON", "AGENT", "REWARD", "ACTION", "POLICY", "STATE", "GRID", "LEARN",
        "TENSOR", "MODEL", "FRAME", "SPACE", "VECTOR", "SOLVE", "TRAIN", "GYM",
        "GAME", "CODE", "PLAY", "TEST", "LOOP", "DATA", "STEP", "DONE", "INFO"
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
        
        self.font_letter = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.font_ui_small = pygame.font.SysFont("Arial", 16)
        
        # State variables
        self.grid = []
        self.cursor_pos = [0, 0]
        self.visual_cursor_pos = [0, 0] # For smooth interpolation
        self.current_selection = [] # list of (x, y) grid coords
        self.hidden_words_data = {} # word_str -> {coords, found}
        self.found_words = set()
        self.steps = 0
        self.score = 0
        self.timer = 0.0
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.feedback_flash = {"color": (0,0,0,0), "alpha": 0}
        
        self.reset()
        
        self.validate_implementation()
    
    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.hidden_words_data = {}
        
        words_to_place = random.sample(self.WORD_LIST, 10)
        
        for word in words_to_place:
            placed = False
            for _ in range(100): # 100 attempts to place a word
                direction = random.choice([(1, 0), (0, 1), (1, 1), (-1, 1)])
                
                if direction[0] == 1:
                    start_x = random.randint(0, self.GRID_COLS - len(word))
                elif direction[0] == -1:
                    start_x = random.randint(len(word) - 1, self.GRID_COLS - 1)
                else:
                    start_x = random.randint(0, self.GRID_COLS - 1)
                    
                if direction[1] == 1:
                    start_y = random.randint(0, self.GRID_ROWS - len(word))
                else:
                    start_y = random.randint(0, self.GRID_ROWS - 1)

                x, y = start_x, start_y
                
                # Check for collisions
                can_place = True
                coords = []
                for char in word:
                    if not (0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS):
                        can_place = False
                        break
                    if self.grid[y][x] != '' and self.grid[y][x] != char:
                        can_place = False
                        break
                    coords.append((x, y))
                    x += direction[0]
                    y += direction[1]
                
                if can_place:
                    x, y = start_x, start_y
                    for char in word:
                        self.grid[y][x] = char
                        x += direction[0]
                        y += direction[1]
                    self.hidden_words_data[word] = {"coords": coords, "found": False}
                    placed = True
                    break
        
        # Fill empty spaces with random letters
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r][c] == '':
                    self.grid[r][c] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.timer = self.MAX_STEPS
        self.game_over = False
        
        self._generate_grid()
        
        self.found_words = set()
        self.current_selection = []
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.visual_cursor_pos = [
            self.GRID_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2,
            self.GRID_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        ]

        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.feedback_flash = {"color": (0,0,0,0), "alpha": 0}
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        
        # --- Handle Input ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Movement
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right
        
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)
        
        # Space press (select letter)
        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            # #SFX: Select letter
            pos = tuple(self.cursor_pos)
            if not self.current_selection:
                self.current_selection.append(pos)
                reward += 0.1
            elif pos not in self.current_selection:
                last_pos = self.current_selection[-1]
                # Check for adjacency
                if abs(pos[0] - last_pos[0]) <= 1 and abs(pos[1] - last_pos[1]) <= 1:
                    self.current_selection.append(pos)
                    reward += 0.1
                else: # Invalid selection (not adjacent)
                    reward -= 0.1
                    self.current_selection = [] # Reset selection on invalid move
                    # #SFX: Invalid selection
                    self._trigger_feedback_flash(self.COLOR_FEEDBACK_BAD)
        
        # Shift press (submit word)
        shift_pressed = shift_held and not self.last_shift_held
        if shift_pressed and self.current_selection:
            word_str = "".join([self.grid[y][x] for x, y in self.current_selection])
            rev_word_str = word_str[::-1]

            found_match = False
            for w, data in self.hidden_words_data.items():
                if not data["found"] and (word_str == w or rev_word_str == w):
                    # Correct word
                    # #SFX: Correct word found
                    data["found"] = True
                    self.found_words.add(w)
                    self.score += 100
                    reward += 10
                    self._trigger_feedback_flash(self.COLOR_FEEDBACK_GOOD)
                    for x, y in self.current_selection:
                        self._spawn_particles(x, y)
                    found_match = True
                    break
            
            if not found_match:
                # Incorrect word
                # #SFX: Incorrect word
                self.score -= 10
                reward -= 1
                self.timer -= self.FPS * 2 # 2 second penalty
                self._trigger_feedback_flash(self.COLOR_FEEDBACK_BAD)
            
            self.current_selection = []

        self.last_space_held, self.last_shift_held = space_held, shift_held
        
        # --- Update Game State ---
        self.steps += 1
        self.timer -= 1
        
        terminated = self._check_termination()
        
        if terminated:
            if len(self.found_words) == len(self.hidden_words_data):
                # Win condition
                self.score += 500
                reward += 50
            else: # Timeout or max steps
                reward -= 10
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _check_termination(self):
        if len(self.found_words) == len(self.hidden_words_data):
            return True # Win
        if self.timer <= 0:
            return True # Timeout
        if self.steps >= self.MAX_STEPS:
            return True # Max steps
        return False
        
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "words_found": len(self.found_words),
            "time_left_seconds": max(0, self.timer / self.FPS),
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw grid background and lines ---
        grid_width_px = self.GRID_COLS * self.CELL_SIZE
        grid_height_px = self.GRID_ROWS * self.CELL_SIZE
        pygame.draw.rect(self.screen, self.COLOR_GRID_LINES, 
                         (self.GRID_X_OFFSET, self.GRID_Y_OFFSET, grid_width_px, grid_height_px), 1)

        # --- Draw letters and found word backgrounds ---
        found_coords = set()
        for word, data in self.hidden_words_data.items():
            if data["found"]:
                for coord in data["coords"]:
                    found_coords.add(tuple(coord))

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                char = self.grid[r][c]
                px = self.GRID_X_OFFSET + c * self.CELL_SIZE
                py = self.GRID_Y_OFFSET + r * self.CELL_SIZE
                
                if (c, r) in found_coords:
                    pygame.draw.rect(self.screen, self.COLOR_FOUND_WORD_BG, (px, py, self.CELL_SIZE, self.CELL_SIZE))
                    letter_surf = self.font_letter.render(char, True, self.COLOR_FOUND_WORD_FG)
                else:
                    letter_surf = self.font_letter.render(char, True, self.COLOR_LETTER)
                
                text_rect = letter_surf.get_rect(center=(px + self.CELL_SIZE / 2, py + self.CELL_SIZE / 2))
                self.screen.blit(letter_surf, text_rect)
    
        # --- Draw selection line ---
        if len(self.current_selection) > 1:
            points = []
            for x, y in self.current_selection:
                points.append((
                    self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE / 2,
                    self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE / 2
                ))
            pygame.draw.lines(self.screen, self.COLOR_SELECTION, False, points, 5)

        # --- Draw selection circles ---
        for x, y in self.current_selection:
            center_x = int(self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE / 2)
            center_y = int(self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE / 2)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(self.CELL_SIZE * 0.45), self.COLOR_SELECTION)
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, int(self.CELL_SIZE * 0.45) - 1, self.COLOR_SELECTION)
        
        # --- Interpolate and draw cursor ---
        target_x = self.GRID_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE + self.CELL_SIZE / 2
        target_y = self.GRID_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE + self.CELL_SIZE / 2
        self.visual_cursor_pos[0] += (target_x - self.visual_cursor_pos[0]) * 0.5
        self.visual_cursor_pos[1] += (target_y - self.visual_cursor_pos[1]) * 0.5
        
        cursor_rect = pygame.Rect(0, 0, self.CELL_SIZE, self.CELL_SIZE)
        cursor_rect.center = (self.visual_cursor_pos[0], self.visual_cursor_pos[1])
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

        # --- Update and draw particles ---
        self._update_and_draw_particles()
        
        # --- Draw feedback flash ---
        if self.feedback_flash["alpha"] > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            flash_surface.fill(self.feedback_flash["color"])
            self.screen.blit(flash_surface, (0, 0))
            self.feedback_flash["alpha"] -= 15
            self.feedback_flash["color"] = (*self.feedback_flash["color"][:3], max(0, self.feedback_flash["alpha"]))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render("SCORE", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (40, 20))
        score_val = self.font_ui.render(f"{self.score}", True, self.COLOR_UI_VALUE)
        self.screen.blit(score_val, (40, 40))

        # Time
        time_left_sec = max(0, self.timer / self.FPS)
        time_color = self.COLOR_UI_VALUE if time_left_sec > 10 else self.COLOR_TIMER_WARN
        time_text = self.font_ui.render("TIME", True, self.COLOR_UI_TEXT)
        time_text_rect = time_text.get_rect(centerx=self.SCREEN_WIDTH / 2)
        time_text_rect.top = 20
        self.screen.blit(time_text, time_text_rect)
        time_val = self.font_ui.render(f"{time_left_sec:.1f}", True, time_color)
        time_val_rect = time_val.get_rect(centerx=self.SCREEN_WIDTH / 2)
        time_val_rect.top = 40
        self.screen.blit(time_val, time_val_rect)

        # Words Found
        words_text = self.font_ui.render("WORDS", True, self.COLOR_UI_TEXT)
        words_text_rect = words_text.get_rect(right=self.SCREEN_WIDTH - 40, top=20)
        self.screen.blit(words_text, words_text_rect)
        words_val = self.font_ui.render(f"{len(self.found_words)} / {len(self.hidden_words_data)}", True, self.COLOR_UI_VALUE)
        words_val_rect = words_val.get_rect(right=self.SCREEN_WIDTH - 40, top=40)
        self.screen.blit(words_val, words_val_rect)

    def _spawn_particles(self, grid_x, grid_y):
        center_x = self.GRID_X_OFFSET + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = self.GRID_Y_OFFSET + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = random.randint(15, 30)
            color = random.choice([self.COLOR_FOUND_WORD_FG, (200, 255, 220), self.COLOR_SELECTION])
            self.particles.append([
                [center_x, center_y], # pos
                vel, # vel
                life,
                color
            ])
            
    def _update_and_draw_particles(self):
        for p in self.particles:
            p[0][0] += p[1][0] # pos.x += vel.x
            p[0][1] += p[1][1] # pos.y += vel.y
            p[2] -= 1 # life -= 1
            
            radius = max(0, p[2] / 6)
            pygame.draw.circle(self.screen, p[3], p[0], radius)
        
        self.particles = [p for p in self.particles if p[2] > 0]

    def _trigger_feedback_flash(self, color):
        self.feedback_flash["color"] = color
        self.feedback_flash["alpha"] = color[3]

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

# Example usage to test the environment visually
if __name__ == '__main__':
    # To run with visualization, unset the dummy driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Word Search")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
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
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000)

        clock.tick(GameEnv.FPS)
        
    env.close()