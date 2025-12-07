
# Generated: 2025-08-28T05:07:27.365009
# Source Brief: brief_05466.md
# Brief Index: 5466

        
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
        "Use arrows to move the cursor. Press space on the first letter of a word, "
        "move to the last letter, and press shift to submit."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Find all the hidden words in the grid before the timer runs out. "
        "Words can be horizontal, vertical, or diagonal."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 10
        self.CELL_SIZE = 32
        self.GRID_ORIGIN_X = (self.WIDTH - self.GRID_SIZE * self.CELL_SIZE) // 2
        self.GRID_ORIGIN_Y = (self.HEIGHT - self.GRID_SIZE * self.CELL_SIZE) // 2 + 20
        self.GAME_DURATION_SECONDS = 120
        self.FPS = 30
        self.MAX_STEPS = self.GAME_DURATION_SECONDS * self.FPS
        self.NUM_WORDS = 5

        # Colors
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_LINE = (50, 60, 70)
        self.COLOR_LETTER = (220, 220, 230)
        self.COLOR_CURSOR = (255, 200, 0)
        self.COLOR_SELECTION = (255, 200, 0, 100) # RGBA for transparency
        self.COLOR_FOUND_WORD_BG = (40, 120, 80, 150)
        self.COLOR_FOUND_WORD_STRIKE = (180, 255, 200)
        self.COLOR_UI_TEXT = (200, 200, 220)
        self.COLOR_TIMER_LOW = (255, 80, 80)
        self.PARTICLE_COLORS = [(255, 200, 0), (255, 255, 100), (250, 150, 50)]

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
        self.font_grid = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_ui = pygame.font.SysFont("tahoma", 18)
        self.font_ui_small = pygame.font.SysFont("tahoma", 14)

        # Word list
        self.WORD_BANK = [
            "PYTHON", "AGENT", "REWARD", "STATE", "ACTION", "POLICY", "GRID",
            "LEARN", "MODEL", "SOLVE", "SEARCH", "PUZZLE", "TIMER", "SCORE",
            "SPACE", "VECTOR", "TENSOR", "ALPHA", "BETA", "GAMMA", "DELTA",
            "EPOCH", "BATCH", "TRAIN", "VALID"
        ]

        # Etc...
        self.grid = None
        self.words_to_find = None
        self.word_solutions = None
        self.found_words_data = None
        self.cursor_pos = None
        self.selection_active = None
        self.selection_start_pos = None
        self.selection_path = None
        self.timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.particles = None
        self.prev_space_held = None
        self.prev_shift_held = None
        self.reward_this_step = 0
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.timer = self.GAME_DURATION_SECONDS
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selection_active = False
        self.selection_start_pos = None
        self.selection_path = []
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_new_puzzle()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()

    def _generate_new_puzzle(self):
        # Retry until a valid puzzle is generated
        while True:
            self.grid = np.full((self.GRID_SIZE, self.GRID_SIZE), "_", dtype=str)
            self.words_to_find = self.np_random.choice(self.WORD_BANK, self.NUM_WORDS, replace=False).tolist()
            self.word_solutions = {}
            self.found_words_data = []
            
            words_placed = 0
            for word in self.words_to_find:
                if self._place_word(word):
                    words_placed += 1
            
            if words_placed == self.NUM_WORDS:
                break # Success
        
        # Fill empty cells with random letters
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if self.grid[r, c] == "_":
                    self.grid[r, c] = chr(self.np_random.integers(65, 91)) # A-Z

    def _place_word(self, word):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
        self.np_random.shuffle(directions)
        
        for _ in range(50): # 50 placement attempts
            direction = directions[self.np_random.integers(0, len(directions))]
            dr, dc = direction
            
            start_r = self.np_random.integers(0, self.GRID_SIZE)
            start_c = self.np_random.integers(0, self.GRID_SIZE)
            
            end_r = start_r + (len(word) - 1) * dr
            end_c = start_c + (len(word) - 1) * dc
            
            if not (0 <= end_r < self.GRID_SIZE and 0 <= end_c < self.GRID_SIZE):
                continue # Word goes out of bounds

            can_place = True
            path = []
            for i in range(len(word)):
                r, c = start_r + i * dr, start_c + i * dc
                if self.grid[r, c] != '_' and self.grid[r, c] != word[i]:
                    can_place = False
                    break
                path.append((r, c))

            if can_place:
                for i in range(len(word)):
                    r, c = path[i]
                    self.grid[r, c] = word[i]
                self.word_solutions[word] = path
                self.word_solutions[word[::-1]] = path[::-1] # Also accept reversed word
                return True
        return False

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.reward_this_step = 0
        self.steps += 1
        
        # Update timer
        self.timer -= 1 / self.FPS
        
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        is_space_press = space_held and not self.prev_space_held
        is_shift_press = shift_held and not self.prev_shift_held
        
        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[0] -= 1  # Up
        elif movement == 2: self.cursor_pos[0] += 1  # Down
        elif movement == 3: self.cursor_pos[1] -= 1  # Left
        elif movement == 4: self.cursor_pos[1] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_SIZE - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_SIZE - 1)

        # 2. Handle selection logic
        if is_space_press and not self.selection_active:
            # Start a new selection
            self.selection_active = True
            self.selection_start_pos = tuple(self.cursor_pos)
            self.selection_path = [self.selection_start_pos]
            # sfx: click_start
            
        if self.selection_active:
            # Update selection path based on cursor
            self._update_selection_path()
        
        if is_shift_press and self.selection_active:
            # Submit word
            self._submit_word()
            # sfx: submit_word
            self.selection_active = False
            self.selection_start_pos = None
            self.selection_path = []

        # 3. Update game state
        self._update_particles()
        
        # 4. Check for termination
        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if len(self.found_words_data) == self.NUM_WORDS:
                self.reward_this_step += 50 # Victory bonus
                # sfx: win_game

        # Update previous action states
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        if self.auto_advance:
            self.clock.tick(self.FPS)
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _update_selection_path(self):
        if not self.selection_start_pos: return
        
        path = []
        start_r, start_c = self.selection_start_pos
        end_r, end_c = self.cursor_pos
        
        dr, dc = end_r - start_r, end_c - start_c
        
        # Only allow straight lines (horizontal, vertical, diagonal)
        if dr == 0 or dc == 0 or abs(dr) == abs(dc):
            steps = max(abs(dr), abs(dc))
            r_step = np.sign(dr) if dr != 0 else 0
            c_step = np.sign(dc) if dc != 0 else 0
            
            for i in range(steps + 1):
                path.append((start_r + i * r_step, start_c + i * c_step))
        
        self.selection_path = path

    def _submit_word(self):
        if not self.selection_path: return
        
        selected_word = "".join([self.grid[r,c] for r,c in self.selection_path])
        found_words_list = [data['word'] for data in self.found_words_data]

        if selected_word in self.word_solutions and selected_word not in found_words_list:
            # Correct word found!
            self.score += 10
            self.reward_this_step += 10
            path = self.word_solutions[selected_word]
            self.found_words_data.append({'word': selected_word, 'path': path})
            self._create_word_found_particles(path)
            # sfx: success
        else:
            # Incorrect submission
            self.reward_this_step -= 1
            # sfx: failure

    def _create_word_found_particles(self, path):
        # Find center of the word
        center_r = sum(r for r, c in path) / len(path)
        center_c = sum(c for r, c in path) / len(path)
        center_x = self.GRID_ORIGIN_X + center_c * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = self.GRID_ORIGIN_Y + center_r * self.CELL_SIZE + self.CELL_SIZE / 2
        
        for _ in range(30):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(20, 40)
            radius = self.np_random.uniform(2, 5)
            color = random.choice(self.PARTICLE_COLORS)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifespan, 'max_life': lifespan, 'radius': radius, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _check_termination(self):
        return self.timer <= 0 or len(self.found_words_data) == self.NUM_WORDS or self.steps >= self.MAX_STEPS
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
            "words_found": len(self.found_words_data),
        }

    def _render_game(self):
        # Draw grid lines
        for i in range(self.GRID_SIZE + 1):
            # Vertical
            start_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y)
            end_pos = (self.GRID_ORIGIN_X + i * self.CELL_SIZE, self.GRID_ORIGIN_Y + self.GRID_SIZE * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)
            # Horizontal
            start_pos = (self.GRID_ORIGIN_X, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            end_pos = (self.GRID_ORIGIN_X + self.GRID_SIZE * self.CELL_SIZE, self.GRID_ORIGIN_Y + i * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start_pos, end_pos, 1)

        # Draw found words background
        for data in self.found_words_data:
            for r, c in data['path']:
                rect = pygame.Rect(
                    self.GRID_ORIGIN_X + c * self.CELL_SIZE,
                    self.GRID_ORIGIN_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                s.fill(self.COLOR_FOUND_WORD_BG)
                self.screen.blit(s, rect.topleft)

        # Draw letters
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                letter = self.grid[r, c]
                text_surf = self.font_grid.render(letter, True, self.COLOR_LETTER)
                text_rect = text_surf.get_rect(center=(
                    self.GRID_ORIGIN_X + c * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.GRID_ORIGIN_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
                ))
                self.screen.blit(text_surf, text_rect)

        # Draw selection highlight
        if self.selection_active and self.selection_path:
            for r, c in self.selection_path:
                s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
                s.fill(self.COLOR_SELECTION)
                self.screen.blit(s, (self.GRID_ORIGIN_X + c * self.CELL_SIZE, self.GRID_ORIGIN_Y + r * self.CELL_SIZE))
                
        # Draw cursor
        cursor_r, cursor_c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_ORIGIN_X + cursor_c * self.CELL_SIZE,
            self.GRID_ORIGIN_Y + cursor_r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=4)
        
        # Draw found words strikethrough
        for data in self.found_words_data:
            start_r, start_c = data['path'][0]
            end_r, end_c = data['path'][-1]
            start_pos = (
                self.GRID_ORIGIN_X + start_c * self.CELL_SIZE + self.CELL_SIZE // 2,
                self.GRID_ORIGIN_Y + start_r * self.CELL_SIZE + self.CELL_SIZE // 2
            )
            end_pos = (
                self.GRID_ORIGIN_X + end_c * self.CELL_SIZE + self.CELL_SIZE // 2,
                self.GRID_ORIGIN_Y + end_r * self.CELL_SIZE + self.CELL_SIZE // 2
            )
            pygame.draw.line(self.screen, self.COLOR_FOUND_WORD_STRIKE, start_pos, end_pos, 4)

        # Draw particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            radius = int(p['radius'] * (p['life'] / p['max_life']))
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)
                pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)


    def _render_ui(self):
        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.font_ui.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (20, 10))

        # Timer
        timer_color = self.COLOR_UI_TEXT if self.timer > 10 else self.COLOR_TIMER_LOW
        timer_text = f"TIME: {max(0, int(self.timer))}"
        timer_surf = self.font_ui.render(timer_text, True, timer_color)
        timer_rect = timer_surf.get_rect(center=(self.WIDTH // 2, 10 + timer_surf.get_height() // 2))
        self.screen.blit(timer_surf, timer_rect)

        # Words to find
        words_x_start = self.WIDTH - 150
        found_words_list = [data['word'] for data in self.found_words_data]
        for i, word in enumerate(self.words_to_find):
            is_found = word in found_words_list
            color = self.COLOR_FOUND_WORD_STRIKE if is_found else self.COLOR_UI_TEXT
            word_surf = self.font_ui_small.render(word, True, color)
            self.screen.blit(word_surf, (words_x_start, 10 + i * 18))
            if is_found:
                line_y = 10 + i * 18 + word_surf.get_height() // 2
                pygame.draw.line(self.screen, color, (words_x_start, line_y), (words_x_start + word_surf.get_width(), line_y), 2)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0,0))
            
            msg = "YOU WIN!" if len(self.found_words_data) == self.NUM_WORDS else "TIME'S UP!"
            msg_surf = self.font_grid.render(msg, True, (255, 255, 255))
            msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
            self.screen.blit(msg_surf, msg_rect)

    def close(self):
        pygame.font.quit()
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
    # This block allows you to run the file directly to see and play the game.
    # It will not be run when the class is imported.
    import os
    # Comment out the line below to run with a visible window
    # os.environ["SDL_VIDEODRIVER"] = "dummy" 
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Pygame setup for human play ---
    if "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy":
        pygame.display.set_caption("Word Search Environment")
        real_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))

    obs, info = env.reset()
    terminated = False
    
    # Game loop for human play
    running = True
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Reward: {reward:.2f}, Score: {info['score']}, Words Found: {info['words_found']}")

            if terminated:
                print(f"Episode finished. Final Info: {info}")
        
        # --- Rendering for human play ---
        if "SDL_VIDEODRIVER" not in os.environ or os.environ["SDL_VIDEODRIVER"] != "dummy":
            # The observation is (H, W, C), but pygame needs (W, H) surface
            # We already have the screen surface in the env, so we can just use that
            surf = pygame.transform.flip(env.screen, False, True)
            real_screen.blit(surf, (0, 0))
            pygame.display.flip()

    env.close()