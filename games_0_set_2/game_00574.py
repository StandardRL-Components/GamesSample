
# Generated: 2025-08-27T14:03:36.930159
# Source Brief: brief_00574.md
# Brief Index: 574

        
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
        "Controls: Use arrow keys to move the worm. Press Space to submit the current word."
    )

    game_description = (
        "Guide a worm to connect letters and form words. Find 5 valid words before you run out of moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_COLS, self.GRID_ROWS = 10, 8
        self.GRID_MARGIN_X = (self.WIDTH - 400) // 2
        self.GRID_MARGIN_Y = 20
        self.CELL_SIZE = 40
        self.MAX_MOVES = 50
        self.WIN_CONDITION = 5
        self.MIN_WORD_LEN = 3

        # --- Colors ---
        self.COLOR_BG = (20, 30, 40)
        self.COLOR_GRID_BG = (30, 45, 60)
        self.COLOR_GRID_LINE = (50, 65, 80)
        self.COLOR_LETTER = (200, 210, 220)
        self.COLOR_WORM = (50, 220, 150)
        self.COLOR_WORM_OUTLINE = (30, 180, 120)
        self.COLOR_HIGHLIGHT = (255, 200, 50, 150) # With alpha
        self.COLOR_UI_TEXT = (220, 230, 240)
        self.COLOR_SUCCESS = (100, 255, 150)
        self.COLOR_FAIL = (255, 100, 100)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        # --- Word Dictionary ---
        self._generate_word_list()

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.worm_path = None
        self.worm_head = None
        self.start_of_word_pos = None
        self.current_word = None
        self.completed_words = None
        self.moves_remaining = None
        self.score = None
        self.game_over = None
        self.feedback_message = None
        self.feedback_timer = None
        self.feedback_color = None
        self.space_was_down = None
        self.steps = None
        self.particles = None
        
        self.reset()
        
        # self.validate_implementation() # Optional self-check

    def _generate_word_list(self):
        # In a real scenario, this would be a larger list.
        # For this self-contained example, we embed a small dictionary.
        words = [
            "python", "gym", "code", "agent", "reward", "action", "state", "game",
            "play", "word", "grid", "worm", "move", "win", "loss", "path", "find",
            "list", "api", "env", "step", "reset", "deep", "learn", "mind", "quest",
            "vector", "matrix", "pixel", "render", "train", "policy", "value", "tree",
            "search", "node", "branch", "leaf", "apple", "grape", "kiwi", "lime"
        ]
        self.WORD_SET = {w.upper() for w in words if len(w) >= self.MIN_WORD_LEN}
        self.PREFIX_SET = set()
        for word in self.WORD_SET:
            for i in range(1, len(word) + 1):
                self.PREFIX_SET.add(word[:i])

    def _generate_grid(self):
        grid = [['' for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        
        # Place a few guaranteed words
        words_to_place = self.np_random.choice(list(self.WORD_SET), size=3, replace=False)
        for word in words_to_place:
            placed = False
            for _ in range(20): # Try 20 times to place a word
                direction = self.np_random.choice(['h', 'v'])
                if direction == 'h':
                    r = self.np_random.integers(0, self.GRID_ROWS)
                    c = self.np_random.integers(0, self.GRID_COLS - len(word))
                    if all(grid[r][c+i] == '' or grid[r][c+i] == word[i] for i in range(len(word))):
                        for i, char in enumerate(word):
                            grid[r][c+i] = char
                        placed = True
                        break
                else: # vertical
                    r = self.np_random.integers(0, self.GRID_ROWS - len(word))
                    c = self.np_random.integers(0, self.GRID_COLS)
                    if all(grid[r+i][c] == '' or grid[r+i][c] == word[i] for i in range(len(word))):
                        for i, char in enumerate(word):
                            grid[r+i][c] = char
                        placed = True
                        break
        
        # Fill remaining spots with random letters, biased towards vowels
        letters = "AAAAAEEEEIIIIIOOOOUU" + "BCDFGHJKLMNPQRSTVWXYZ"
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if grid[r][c] == '':
                    grid[r][c] = self.np_random.choice(list(letters))
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_grid()
        
        start_r = self.np_random.integers(0, self.GRID_ROWS)
        start_c = self.np_random.integers(0, self.GRID_COLS)
        
        self.worm_head = (start_r, start_c)
        self.worm_path = [self.worm_head]
        self.start_of_word_pos = self.worm_head
        self.current_word = self.grid[start_r][start_c]
        
        self.completed_words = []
        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.feedback_message = ""
        self.feedback_timer = 0
        self.space_was_down = False
        self.steps = 0
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
            
        self.steps += 1
        reward = 0
        terminated = False
        
        movement = action[0]
        space_pressed = action[1] == 1
        
        # --- Handle Movement ---
        if movement != 0:
            dr = [-1, 1, 0, 0]
            dc = [0, 0, -1, 1]
            
            # Map action to change in row/col (1=up, 2=down, 3=left, 4=right)
            r, c = self.worm_head
            nr, nc = r + dr[movement-1], c + dc[movement-1]
            
            # Check for valid move
            if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and (nr, nc) not in self.worm_path:
                self.worm_head = (nr, nc)
                self.worm_path.append(self.worm_head)
                self.current_word += self.grid[nr][nc]
                self.moves_remaining -= 1
                
                # Prefix-based reward
                if self.current_word in self.PREFIX_SET:
                    reward += 0.1 # Encourages forming valid prefixes
                else:
                    reward -= 0.1 # Penalizes deviating from valid words
        
        # --- Handle Word Submission ---
        if space_pressed and not self.space_was_down:
            if len(self.current_word) >= self.MIN_WORD_LEN and self.current_word in self.WORD_SET and self.current_word not in self.completed_words:
                # Valid word
                self.completed_words.append(self.current_word)
                word_len_reward = max(0, len(self.current_word) - (self.MIN_WORD_LEN - 1))
                reward += word_len_reward * word_len_reward # Exponential reward for longer words
                self.score += len(self.current_word) * 10
                self._set_feedback(f"'{self.current_word}' found!", self.COLOR_SUCCESS, 60)
                # Create particles for visual effect
                self._create_particles(self.worm_path)
                # Reset for next word
                self.worm_path = [self.worm_head]
                self.start_of_word_pos = self.worm_head
                self.current_word = self.grid[self.worm_head[0]][self.worm_head[1]]
                # sfx: success_chime.wav
            else:
                # Invalid word or too short
                if len(self.current_word) > 1:
                    reward -= 2 # Penalty for submitting an invalid word
                    self._set_feedback("Invalid Word", self.COLOR_FAIL, 45)
                    # Reset path to the start of this attempt
                    self.worm_head = self.start_of_word_pos
                    self.worm_path = [self.worm_head]
                    self.current_word = self.grid[self.worm_head[0]][self.worm_head[1]]
                    # sfx: error_buzz.wav
        
        self.space_was_down = space_pressed
        
        # --- Update feedback timer and particles ---
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
        self._update_particles()
            
        # --- Check for Termination ---
        if len(self.completed_words) >= self.WIN_CONDITION:
            reward += 50
            terminated = True
            self.game_over = True
            self._set_feedback("YOU WIN!", self.COLOR_SUCCESS, 180)
        elif self.moves_remaining <= 0:
            reward -= 50
            terminated = True
            self.game_over = True
            self._set_feedback("Out of Moves!", self.COLOR_FAIL, 180)
        elif self._is_stuck():
            reward -= 50
            terminated = True
            self.game_over = True
            self._set_feedback("Stuck!", self.COLOR_FAIL, 180)
        elif self.steps >= 1000:
            terminated = True # Episode length limit
            self.game_over = True
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _set_feedback(self, message, color, duration):
        self.feedback_message = message
        self.feedback_color = color
        self.feedback_timer = duration

    def _is_stuck(self):
        r, c = self.worm_head
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and (nr, nc) not in self.worm_path:
                return False
        return True

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
            "moves_remaining": self.moves_remaining,
            "completed_words": len(self.completed_words),
        }
        
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_MARGIN_X, self.GRID_MARGIN_Y, self.GRID_COLS * self.CELL_SIZE, self.GRID_ROWS * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect, border_radius=5)
        
        # Highlight current path
        highlight_surface = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        for r, c in self.worm_path:
            px, py = self.GRID_MARGIN_X + c * self.CELL_SIZE, self.GRID_MARGIN_Y + r * self.CELL_SIZE
            pygame.draw.rect(highlight_surface, self.COLOR_HIGHLIGHT, (0, 0, self.CELL_SIZE, self.CELL_SIZE), border_radius=8)
            self.screen.blit(highlight_surface, (px, py))
            
        # Draw letters and grid lines
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                px, py = self.GRID_MARGIN_X + c * self.CELL_SIZE, self.GRID_MARGIN_Y + r * self.CELL_SIZE
                
                # Draw letter
                letter_surf = self.font_large.render(self.grid[r][c], True, self.COLOR_LETTER)
                letter_rect = letter_surf.get_rect(center=(px + self.CELL_SIZE / 2, py + self.CELL_SIZE / 2))
                self.screen.blit(letter_surf, letter_rect)
        
        # Draw worm
        if self.worm_path:
            for i, pos in enumerate(self.worm_path):
                r, c = pos
                px = self.GRID_MARGIN_X + c * self.CELL_SIZE + self.CELL_SIZE / 2
                py = self.GRID_MARGIN_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2
                radius = int(self.CELL_SIZE * 0.3 * (0.8 + 0.2 * (i / len(self.worm_path))))

                # Draw outline
                pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius, self.COLOR_WORM_OUTLINE)
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius, self.COLOR_WORM_OUTLINE)
                
                # Draw body
                pygame.gfxdraw.filled_circle(self.screen, int(px), int(py), radius - 2, self.COLOR_WORM)
                pygame.gfxdraw.aacircle(self.screen, int(px), int(py), radius - 2, self.COLOR_WORM)
                
                # Connect segments
                if i > 0:
                    prev_r, prev_c = self.worm_path[i-1]
                    prev_px = self.GRID_MARGIN_X + prev_c * self.CELL_SIZE + self.CELL_SIZE / 2
                    prev_py = self.GRID_MARGIN_Y + prev_r * self.CELL_SIZE + self.CELL_SIZE / 2
                    pygame.draw.line(self.screen, self.COLOR_WORM_OUTLINE, (int(px), int(py)), (int(prev_px), int(prev_py)), 2 * radius)
                    pygame.draw.line(self.screen, self.COLOR_WORM, (int(px), int(py)), (int(prev_px), int(prev_py)), 2 * (radius-2))

        # Draw particles
        for p in self.particles:
            p_color = list(self.COLOR_SUCCESS)
            p_color.append(p[4]) # Add alpha
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), int(p[3]), p_color)

    def _render_ui(self):
        # --- Top UI (Score, Moves) ---
        score_text = f"Score: {self.score}"
        score_surf = self.font_medium.render(score_text, True, self.COLOR_UI_TEXT)
        self.screen.blit(score_surf, (15, 10))
        
        moves_text = f"Moves: {self.moves_remaining}"
        moves_surf = self.font_medium.render(moves_text, True, self.COLOR_UI_TEXT)
        moves_rect = moves_surf.get_rect(topright=(self.WIDTH - 15, 10))
        self.screen.blit(moves_surf, moves_rect)
        
        # --- Current Word Display ---
        grid_bottom = self.GRID_MARGIN_Y + self.GRID_ROWS * self.CELL_SIZE
        word_bg_rect = pygame.Rect(self.GRID_MARGIN_X, grid_bottom + 5, self.GRID_COLS * self.CELL_SIZE, 30)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, word_bg_rect, border_radius=5)
        
        current_word_surf = self.font_medium.render(self.current_word, True, self.COLOR_HIGHLIGHT)
        current_word_rect = current_word_surf.get_rect(center=word_bg_rect.center)
        self.screen.blit(current_word_surf, current_word_rect)

        # --- Completed Words List ---
        list_x = self.GRID_MARGIN_X + self.GRID_COLS * self.CELL_SIZE + 20
        list_y = self.GRID_MARGIN_Y
        
        title_surf = self.font_medium.render(f"Words ({len(self.completed_words)}/{self.WIN_CONDITION})", True, self.COLOR_UI_TEXT)
        self.screen.blit(title_surf, (list_x, list_y))
        
        for i, word in enumerate(self.completed_words):
            word_surf = self.font_small.render(word, True, self.COLOR_SUCCESS)
            self.screen.blit(word_surf, (list_x + 10, list_y + 30 + i * 20))
            
        # --- Feedback Message ---
        if self.feedback_timer > 0:
            alpha = min(255, int(255 * (self.feedback_timer / 30)))
            feedback_surf = self.font_large.render(self.feedback_message, True, self.feedback_color)
            feedback_surf.set_alpha(alpha)
            feedback_rect = feedback_surf.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(feedback_surf, feedback_rect)

    def _create_particles(self, path):
        for r, c in path:
            px = self.GRID_MARGIN_X + c * self.CELL_SIZE + self.CELL_SIZE / 2
            py = self.GRID_MARGIN_Y + r * self.CELL_SIZE + self.CELL_SIZE / 2
            for _ in range(5):
                angle = self.np_random.uniform(0, 2 * math.pi)
                speed = self.np_random.uniform(1, 3)
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed
                radius = self.np_random.uniform(2, 5)
                lifetime = self.np_random.integers(20, 40)
                self.particles.append([px, py, vx, vy, radius, lifetime, lifetime]) # x, y, vx, vy, r, life, max_life

    def _update_particles(self):
        for p in self.particles[:]:
            p[0] += p[2]
            p[1] += p[3]
            p[5] -= 1
            p[4] -= 0.05 # shrink
            p[2] *= 0.98 # friction
            p[3] *= 0.98
            if p[5] <= 0 or p[4] <= 0:
                self.particles.remove(p)

    def validate_implementation(self):
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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Worm Word Grid")
    
    running = True
    total_reward = 0
    
    print(env.user_guide)
    print(env.game_description)

    while running:
        # --- Human Input ---
        movement = 0 # no-op
        space = 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        
        action = [movement, space, 0] # Shift is not used
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0
                print("--- Game Reset ---")

        # --- Environment Step ---
        # Since auto_advance is False, we only step when there's an action or for continuous key presses
        if movement != 0 or space != 0 or env.space_was_down:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"--- Episode Finished ---")
                print(f"Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
                # Game will freeze on termination, press R to reset
        
        # --- Rendering ---
        # The observation is the rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit FPS for human play

    pygame.quit()