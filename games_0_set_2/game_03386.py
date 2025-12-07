
# Generated: 2025-08-27T23:12:27.312242
# Source Brief: brief_03386.md
# Brief Index: 3386

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ↑↓←→ to move the cursor. Press space to select the first and last letter of a word."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find hidden words in a grid before time runs out in this fast-paced word search puzzle."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    GRID_WIDTH, GRID_HEIGHT = 14, 10
    CELL_SIZE = 36
    GRID_X_OFFSET = (640 - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y_OFFSET = (400 - GRID_HEIGHT * CELL_SIZE) + 20
    FPS = 30
    GAME_DURATION_SECONDS = 60
    WORDS_TO_FIND = 10
    MAX_STEPS = GAME_DURATION_SECONDS * FPS + 10 # A little buffer

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 40, 50)
    COLOR_GRID_LINE = (40, 50, 60)
    COLOR_LETTER = (180, 190, 200)
    COLOR_UI_TEXT = (220, 230, 240)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION = (255, 220, 80)
    COLOR_CORRECT = (0, 255, 120)
    COLOR_INCORRECT = (255, 80, 80)
    
    WORD_LIST = [
        "PYTHON", "GYM", "AGENT", "REWARD", "ACTION", "STATE", "POLICY", "LEARN",
        "DEEP", "SEARCH", "GRID", "GAME", "VECTOR", "TENSOR", "CODE", "DEBUG",
        "VISUAL", "PIXEL", "STEP", "RESET", "FRAME", "PLAY", "SOLVE", "TASK",
        "MODEL", "TRAIN", "DATA", "ALGO", "LOOP", "API"
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()

        # Fonts
        try:
            self.font_letter = pygame.font.SysFont("consolas", 24, bold=True)
            self.font_ui = pygame.font.SysFont("sans", 20, bold=True)
            self.font_timer = pygame.font.SysFont("sans", 28, bold=True)
        except pygame.error:
            self.font_letter = pygame.font.Font(None, 28)
            self.font_ui = pygame.font.Font(None, 24)
            self.font_timer = pygame.font.Font(None, 32)
        
        # State variables
        self.grid = []
        self.placed_words = {}
        self.found_words = set()
        self.cursor_pos = [0, 0]
        self.selected_coords = []
        self.timer = 0
        self.score = 0
        self.steps = 0
        self.prev_space_held = False
        self.particles = []
        self.feedback_flash = {"color": None, "alpha": 0}

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS * self.FPS
        
        self._generate_grid()
        self.found_words = set()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.selected_coords = []
        self.prev_space_held = False
        self.particles = []
        self.feedback_flash = {"color": None, "alpha": 0}

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False

        self._handle_input(action)
        reward += self._update_game_state()

        if self.steps >= self.MAX_STEPS:
            terminated = True

        if self.timer <= 0:
            if not self.game_over: # First frame of game over
                reward -= 50 # Penalty for running out of time
                self.game_over = True
            terminated = True

        if len(self.found_words) >= self.WORDS_TO_FIND:
            if not self.game_over: # First frame of win
                reward += 50 # Bonus for winning
                self.game_over = True
            terminated = True
        
        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )
    
    def _handle_input(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Movement ---
        if movement == 1: self.cursor_pos[1] -= 1 # Up
        elif movement == 2: self.cursor_pos[1] += 1 # Down
        elif movement == 3: self.cursor_pos[0] -= 1 # Left
        elif movement == 4: self.cursor_pos[0] += 1 # Right

        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_WIDTH - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_HEIGHT - 1)

        # --- Selection ---
        if space_held and not self.prev_space_held:
            # SFX: select_click.wav
            coord_tuple = tuple(self.cursor_pos)
            if coord_tuple in self.selected_coords:
                self.selected_coords.remove(coord_tuple)
            else:
                self.selected_coords.append(coord_tuple)
            
            if len(self.selected_coords) > 2:
                self.selected_coords.pop(0)

        self.prev_space_held = space_held

    def _update_game_state(self):
        reward = 0
        if not self.game_over:
            self.timer -= 1
        
        # Check for word submission
        if len(self.selected_coords) == 2:
            coord1, coord2 = self.selected_coords
            
            found_match = False
            for word, details in self.placed_words.items():
                if word in self.found_words:
                    continue
                
                start, end = details["coords"]
                if (coord1 == start and coord2 == end) or \
                   (coord1 == end and coord2 == start):
                    # SFX: correct_word.wav
                    self.found_words.add(word)
                    reward += 10
                    self.score += len(word) * 10
                    self._trigger_feedback_flash(self.COLOR_CORRECT)
                    self._create_particles_for_word(details["path"])
                    found_match = True
                    break
            
            if not found_match:
                # SFX: incorrect_selection.wav
                reward -= 1
                self.score = max(0, self.score - 5)
                self._trigger_feedback_flash(self.COLOR_INCORRECT)

            self.selected_coords = [] # Clear selection after check

        # Update particles
        self.particles = [p for p in self.particles if p["life"] > 0]
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1

        # Update feedback flash
        if self.feedback_flash["alpha"] > 0:
            self.feedback_flash["alpha"] -= 15
            if self.feedback_flash["alpha"] < 0:
                self.feedback_flash["alpha"] = 0

        return reward

    def _generate_grid(self):
        self.grid = [['' for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
        self.placed_words = {}
        
        words_to_place = list(self.WORD_LIST)
        self.np_random.shuffle(words_to_place)
        words_to_place = words_to_place[:self.WORDS_TO_FIND]

        for word in words_to_place:
            placed = False
            for _ in range(100): # 100 attempts to place a word
                direction = self.np_random.choice([(0, 1), (1, 0), (1, 1), (0, -1), (-1, 0), (-1, -1), (1, -1), (-1, 1)])
                
                start_x = self.np_random.integers(0, self.GRID_WIDTH)
                start_y = self.np_random.integers(0, self.GRID_HEIGHT)

                end_x = start_x + (len(word) - 1) * direction[0]
                end_y = start_y + (len(word) - 1) * direction[1]

                if not (0 <= end_x < self.GRID_WIDTH and 0 <= end_y < self.GRID_HEIGHT):
                    continue

                can_place = True
                path = []
                for i in range(len(word)):
                    px, py = start_x + i * direction[0], start_y + i * direction[1]
                    path.append((px, py))
                    if self.grid[py][px] != '' and self.grid[py][px] != word[i]:
                        can_place = False
                        break
                
                if can_place:
                    for i, char in enumerate(word):
                        px, py = start_x + i * direction[0], start_y + i * direction[1]
                        self.grid[py][px] = char
                    self.placed_words[word] = {
                        "coords": ( (start_x, start_y), (end_x, end_y) ),
                        "path": path
                    }
                    placed = True
                    break
            
            if not placed:
                # This should be rare with a large enough grid
                print(f"Warning: Could not place word '{word}'")

        # Fill rest of grid with random letters
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == '':
                    self.grid[y][x] = self.np_random.choice(list(alphabet))

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_found_words()
        self._render_selection_and_cursor()
        self._render_particles()
        self._render_ui()
        self._render_feedback_flash()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_grid(self):
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=5)

        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                letter = self.grid[y][x]
                text_surf = self.font_letter.render(letter, True, self.COLOR_LETTER)
                text_rect = text_surf.get_rect(center=(
                    self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
                ))
                self.screen.blit(text_surf, text_rect)

    def _render_found_words(self):
        for word in self.found_words:
            details = self.placed_words[word]
            start_coord, end_coord = details["coords"]
            
            start_pos = (
                self.GRID_X_OFFSET + start_coord[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                self.GRID_Y_OFFSET + start_coord[1] * self.CELL_SIZE + self.CELL_SIZE // 2
            )
            end_pos = (
                self.GRID_X_OFFSET + end_coord[0] * self.CELL_SIZE + self.CELL_SIZE // 2,
                self.GRID_Y_OFFSET + end_coord[1] * self.CELL_SIZE + self.CELL_SIZE // 2
            )
            pygame.draw.line(self.screen, self.COLOR_CORRECT, start_pos, end_pos, 4)
            
            # Highlight letters of found word
            for x, y in details["path"]:
                center_pos = (
                    self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2,
                )
                pygame.gfxdraw.filled_circle(self.screen, center_pos[0], center_pos[1], self.CELL_SIZE // 2 - 5, (*self.COLOR_CORRECT, 60))


    def _render_selection_and_cursor(self):
        # Render current selections
        for x, y in self.selected_coords:
            center_pos = (
                self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2,
                self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
            )
            pygame.gfxdraw.filled_circle(self.screen, center_pos[0], center_pos[1], self.CELL_SIZE // 2 - 2, (*self.COLOR_SELECTION, 150))
            pygame.gfxdraw.aacircle(self.screen, center_pos[0], center_pos[1], self.CELL_SIZE // 2 - 2, self.COLOR_SELECTION)

        # Render cursor
        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + cx * self.CELL_SIZE,
            self.GRID_Y_OFFSET + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Pulsing glow effect
        glow_alpha = 100 + 50 * math.sin(self.steps * 0.2)
        glow_color = (*self.COLOR_CURSOR, glow_alpha)
        pygame.draw.rect(self.screen, glow_color, cursor_rect.inflate(6, 6), border_radius=8)
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=5)

    def _render_particles(self):
        for p in self.particles:
            size = int(p["life"] / p["max_life"] * p["size"])
            if size > 0:
                pygame.draw.circle(self.screen, p["color"], (int(p["pos"][0]), int(p["pos"][1])), size)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (15, 15))

        # Words Found
        words_text = self.font_ui.render(f"WORDS: {len(self.found_words)}/{self.WORDS_TO_FIND}", True, self.COLOR_UI_TEXT)
        words_rect = words_text.get_rect(centerx=self.screen.get_width() // 2)
        words_rect.top = 15
        self.screen.blit(words_text, words_rect)

        # Timer
        secs = max(0, self.timer / self.FPS)
        timer_color = self.COLOR_UI_TEXT
        if secs < 10 and not self.game_over:
            # Flashing red timer when low
            if self.steps % self.FPS < self.FPS // 2:
                timer_color = self.COLOR_INCORRECT
        
        timer_surf = self.font_timer.render(f"{secs:.1f}", True, timer_color)
        timer_rect = timer_surf.get_rect(right=self.screen.get_width() - 15, top=10)
        self.screen.blit(timer_surf, timer_rect)

    def _render_feedback_flash(self):
        if self.feedback_flash["alpha"] > 0:
            flash_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            color = (*self.feedback_flash["color"], int(self.feedback_flash["alpha"]))
            flash_surface.fill(color)
            self.screen.blit(flash_surface, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer / self.FPS,
            "words_found": len(self.found_words),
        }
    
    def _trigger_feedback_flash(self, color):
        self.feedback_flash["color"] = color
        self.feedback_flash["alpha"] = 128

    def _create_particles_for_word(self, path):
        for x, y in path:
            for _ in range(5): # 5 particles per letter
                px = self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2
                py = self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                life = self.np_random.integers(15, 30)
                self.particles.append({
                    "pos": [px, py],
                    "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                    "life": life,
                    "max_life": life,
                    "color": self.COLOR_CORRECT,
                    "size": self.np_random.integers(2, 5)
                })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'dummy'
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((640, 400))
    pygame.display.set_caption("Word Search")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Action mapping for human play ---
    # action = [movement, space, shift]
    action = [0, 0, 0] 

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No-op default
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
            
        # Space
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        
        # Shift (unused in this game)
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
            obs, info = env.reset()
            total_reward = 0
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(env.FPS)
        
    env.close()