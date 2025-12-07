
# Generated: 2025-08-28T05:51:54.256620
# Source Brief: brief_05718.md
# Brief Index: 5718

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to select a number. "
        "Select a second number that sums to 10 to make a match. "
        "Press Shift or Space on a selected number to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A minimalist number puzzle. Clear the 4x4 grid by matching pairs of numbers that sum to 10. "
        "You have a limited number of moves, so choose wisely!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    # Colors
    COLOR_BG = (30, 30, 40)
    COLOR_GRID = (60, 60, 70)
    COLOR_CURSOR = (255, 200, 0)
    COLOR_SELECTION = (0, 255, 255)
    COLOR_MATCH = (0, 255, 120)
    COLOR_TEXT_UI = (220, 220, 220)
    COLOR_NUM_LOW = (100, 150, 255) # Color for '1'
    COLOR_NUM_HIGH = (200, 100, 255) # Color for '9'
    
    # Dimensions
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_SIZE = 4
    CELL_SIZE = 80
    GRID_MARGIN_X = (SCREEN_WIDTH - GRID_SIZE * CELL_SIZE) // 2
    GRID_MARGIN_Y = (SCREEN_HEIGHT - GRID_SIZE * CELL_SIZE) // 2

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.number_font = pygame.font.SysFont("Consolas", 48, bold=True)
        self.ui_font = pygame.font.SysFont("Arial", 24)
        self.game_over_font = pygame.font.SysFont("Arial", 64, bold=True)
        
        # Initialize state variables
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def _generate_grid(self):
        """Generates a 4x4 grid that is guaranteed to be solvable and has at least 4 initial matches."""
        while True:
            pairs = [(1, 9), (2, 8), (3, 7), (4, 6), (5, 5)]
            full_list = []
            for _ in range(8): # 8 pairs to fill a 16-cell grid
                idx = self.np_random.integers(len(pairs))
                full_list.extend(pairs[idx])
            
            self.np_random.shuffle(full_list)
            grid = np.array(full_list).reshape((self.GRID_SIZE, self.GRID_SIZE))
            
            # Validate that at least 4 matches are possible
            counts = {i: np.count_nonzero(grid == i) for i in range(1, 10)}
            num_matches = 0
            num_matches += min(counts[1], counts[9])
            num_matches += min(counts[2], counts[8])
            num_matches += min(counts[3], counts[7])
            num_matches += min(counts[4], counts[6])
            num_matches += counts[5] // 2
            
            if num_matches >= 4:
                return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_grid()
        self.steps = 0
        self.score = 0
        self.moves_left = 5
        self.game_over = False
        self.win = False
        
        self.cursor_pos = np.array([0, 0])
        self.first_selection = None
        
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.particles = []

        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Handle Input ---
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held

        # 1. Handle cursor movement
        if movement == 1: self.cursor_pos[0] -= 1 # Up
        elif movement == 2: self.cursor_pos[0] += 1 # Down
        elif movement == 3: self.cursor_pos[1] -= 1 # Left
        elif movement == 4: self.cursor_pos[1] += 1 # Right
        self.cursor_pos = np.clip(self.cursor_pos, 0, self.GRID_SIZE - 1)

        # 2. Handle deselection with Shift
        if shift_pressed and self.first_selection is not None:
            self.first_selection = None
            # No reward/penalty for explicit deselect

        # 3. Handle selection with Space
        if space_pressed:
            r, c = self.cursor_pos
            
            # Case A: A cell is already selected
            if self.first_selection is not None:
                r1, c1 = self.first_selection
                
                if r == r1 and c == c1:
                    self.first_selection = None
                    reward -= 0.1 # Penalty for redundant action
                elif self.grid[r, c] != 0:
                    self.moves_left -= 1
                    num1 = self.grid[r1, c1]
                    num2 = self.grid[r, c]
                    
                    if num1 + num2 == 10:
                        reward += 10
                        self.score += 10
                        self.grid[r1, c1] = 0
                        self.grid[r, c] = 0
                        # sound: "match_success.wav"
                        self._create_particles(c1, r1)
                        self._create_particles(c, r)
                    else:
                        reward -= 1
                        # sound: "match_fail.wav"
                    
                    self.first_selection = None

            # Case B: No cell is selected yet
            elif self.grid[r, c] > 0:
                self.first_selection = (r, c)
                reward += 1 # Reward for valid first selection
                # sound: "select.wav"
        
        # --- Update Game Logic ---
        self.steps += 1
        self._update_particles()
        
        terminated = self._check_termination()
        if terminated:
            if self.win:
                reward += 100
                self.score += 100
                # sound: "win.wav"
            else:
                reward -= 50
                # sound: "lose.wav"

        # Update button states for next step
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _check_termination(self):
        if np.all(self.grid == 0):
            self.win = True
            self.game_over = True
            return True
        elif self.moves_left <= 0:
            self.win = False
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid_and_numbers()
        self._render_particles()
        self._render_cursor_and_selection()
        
        # Render UI overlay
        self._render_ui()

        if self.game_over:
            self._render_game_over_screen()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.moves_left}

    def _get_number_color(self, n):
        fraction = (n - 1) / 8.0 # Normalize 1-9 to 0-1
        r = self.COLOR_NUM_LOW[0] + fraction * (self.COLOR_NUM_HIGH[0] - self.COLOR_NUM_LOW[0])
        g = self.COLOR_NUM_LOW[1] + fraction * (self.COLOR_NUM_HIGH[1] - self.COLOR_NUM_LOW[1])
        b = self.COLOR_NUM_LOW[2] + fraction * (self.COLOR_NUM_HIGH[2] - self.COLOR_NUM_LOW[2])
        return (int(r), int(g), int(b))

    def _draw_text(self, text, font, color, surface, x, y, center=True):
        text_obj = font.render(text, True, color)
        text_rect = text_obj.get_rect()
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
        surface.blit(text_obj, text_rect)

    def _render_grid_and_numbers(self):
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                cell_rect = pygame.Rect(
                    self.GRID_MARGIN_X + c * self.CELL_SIZE,
                    self.GRID_MARGIN_Y + r * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, self.COLOR_GRID, cell_rect, 1)

                num = self.grid[r, c]
                if num > 0:
                    color = self._get_number_color(num)
                    self._draw_text(str(num), self.number_font, color, self.screen, cell_rect.centerx, cell_rect.centery)

    def _render_cursor_and_selection(self):
        if self.first_selection is not None:
            r, c = self.first_selection
            sel_rect = pygame.Rect(
                self.GRID_MARGIN_X + c * self.CELL_SIZE,
                self.GRID_MARGIN_Y + r * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, sel_rect, 4)

        r, c = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_MARGIN_X + c * self.CELL_SIZE,
            self.GRID_MARGIN_Y + r * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 4)

    def _render_ui(self):
        self._draw_text(f"Moves Left: {self.moves_left}", self.ui_font, self.COLOR_TEXT_UI, self.screen, 10, 10, center=False)
        score_text = f"Score: {self.score}"
        score_width = self.ui_font.size(score_text)[0]
        self._draw_text(score_text, self.ui_font, self.COLOR_TEXT_UI, self.screen, self.SCREEN_WIDTH - score_width - 10, 10, center=False)

    def _render_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        
        message = "YOU WIN!" if self.win else "GAME OVER"
        color = self.COLOR_MATCH if self.win else self.COLOR_SELECTION
        self._draw_text(message, self.game_over_font, color, self.screen, self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)

    def _create_particles(self, c, r):
        x = self.GRID_MARGIN_X + c * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.GRID_MARGIN_Y + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(20):
            particle = {
                "x": x, "y": y,
                "vx": random.uniform(-3, 3), "vy": random.uniform(-3, 3),
                "radius": random.uniform(3, 8),
                "life": random.randint(20, 40), "max_life": 40,
            }
            self.particles.append(particle)
    
    def _update_particles(self):
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["life"] -= 1
            p["radius"] -= 0.1
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = p["life"] / p["max_life"]
            alpha = int(255 * life_ratio)
            radius = int(p["radius"])
            if radius <= 0: continue
            
            temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            color = self.COLOR_MATCH + (alpha,)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (int(p["x"] - radius), int(p["y"] - radius)))

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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Number Match Puzzle")
    clock = pygame.time.Clock()
    
    while not terminated:
        movement, space, shift = 0, 0, 0
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
        
        obs, reward, term, truncated, info = env.step(action)
        terminated = term
        
        if reward != 0:
            print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves Left: {info['moves_left']}")

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)
    
    print("Game Over!")
    pygame.time.wait(3000)
    env.close()