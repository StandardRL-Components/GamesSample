
# Generated: 2025-08-27T17:46:38.717173
# Source Brief: brief_01636.md
# Brief Index: 1636

        
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

    # User-facing control string
    user_guide = (
        "Controls: Use arrow keys to move the cursor. "
        "Press space to select a number. Match pairs that sum to 10."
    )

    # User-facing description of the game
    game_description = (
        "A fast-paced puzzle game. Race against the clock to find and "
        "match pairs of numbers that sum to 10."
    )

    # Frames auto-advance for real-time gameplay
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 8, 6
    GRID_X_OFFSET, GRID_Y_OFFSET = 100, 40
    CELL_SIZE = 50
    TARGET_SUM = 10
    MAX_TIME = 90.0  # seconds
    FPS = 30

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID = (40, 60, 80)
    COLOR_TEXT = (220, 230, 240)
    COLOR_NUMBER = (255, 255, 255)
    COLOR_CURSOR = (0, 200, 255)
    COLOR_SELECTION = (255, 200, 0)
    COLOR_SUCCESS = (0, 255, 150)
    COLOR_FAIL = (255, 80, 80)
    COLOR_UI_BG = (30, 45, 60, 200)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Exact spaces as required
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
        self.number_font = pygame.font.SysFont("Consolas", 32, bold=True)
        self.ui_font = pygame.font.SysFont("Consolas", 18, bold=True)
        
        # Initialize state variables
        self.np_random = None
        self.grid = None
        self.cursor_pos = None
        self.selected_cell = None
        self.selected_value = None
        self.time_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.previous_space_held = None
        self.pairs_left = None
        self.particles = None
        self.fading_cells = None
        
        self.reset()
        
        # Run validation check
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_left = self.MAX_TIME
        self.cursor_pos = [0, 0]
        self.selected_cell = None
        self.selected_value = None
        self.previous_space_held = False
        self.particles = []
        self.fading_cells = []
        
        self._generate_grid()
        
        return self._get_observation(), self._get_info()

    def _generate_grid(self):
        """Creates a solvable grid of numbers."""
        num_cells = self.GRID_COLS * self.GRID_ROWS
        num_pairs = num_cells // 2
        self.pairs_left = num_pairs

        possible_pairs = [(i, self.TARGET_SUM - i) for i in range(1, self.TARGET_SUM // 2 + 1)]
        
        numbers = []
        for _ in range(num_pairs):
            pair = possible_pairs[self.np_random.integers(0, len(possible_pairs))]
            numbers.extend(pair)
        
        self.np_random.shuffle(numbers)
        
        self.grid = np.array(numbers).reshape((self.GRID_ROWS, self.GRID_COLS)).tolist()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        
        reward = 0
        self.steps += 1
        self.time_left -= 1.0 / self.FPS

        # --- Game Logic ---
        old_cursor_pos = tuple(self.cursor_pos)
        self._handle_movement(movement)
        
        # Continuous movement reward
        reward += self._calculate_movement_reward(old_cursor_pos, tuple(self.cursor_pos))

        space_pressed = space_held and not self.previous_space_held
        if space_pressed:
            match_reward, feedback_type = self._handle_selection()
            reward += match_reward
            if feedback_type == "success":
                # Sound: Correct match
                pass
            elif feedback_type == "fail":
                # Sound: Incorrect match
                pass

        self.previous_space_held = space_held

        # --- Termination Check ---
        terminated = False
        if self.time_left <= 0:
            terminated = True
            reward -= 100  # Time-out penalty
            # Sound: Game over
        elif self.pairs_left == 0:
            terminated = True
            reward += 100  # Victory bonus
            # Sound: Victory
            
        self.game_over = terminated
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        """Updates cursor position with wrap-around."""
        if movement == 1:  # Up
            self.cursor_pos[1] -= 1
        elif movement == 2:  # Down
            self.cursor_pos[1] += 1
        elif movement == 3:  # Left
            self.cursor_pos[0] -= 1
        elif movement == 4:  # Right
            self.cursor_pos[0] += 1
        
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS

    def _handle_selection(self):
        """Handles logic for selecting numbers and checking for matches."""
        cx, cy = self.cursor_pos
        cell_value = self.grid[cy][cx]
        
        if cell_value is None:
            return 0, None # Clicked an empty cell

        # If nothing is selected, select the current cell
        if self.selected_cell is None:
            self.selected_cell = (cx, cy)
            self.selected_value = cell_value
            # Sound: Select
            return 0, "select"

        # If clicking the same cell, deselect it
        if self.selected_cell == (cx, cy):
            self.selected_cell = None
            self.selected_value = None
            # Sound: Deselect
            return 0, "deselect"
            
        # A second, different cell is selected, check for a match
        if self.selected_value + cell_value == self.TARGET_SUM:
            # --- Success ---
            self.score += 10
            
            # Create fade-out effect for both cells
            sx, sy = self.selected_cell
            self.fading_cells.append({'pos': (sx, sy), 'value': self.grid[sy][sx], 'alpha': 255})
            self.fading_cells.append({'pos': (cx, cy), 'value': self.grid[cy][cx], 'alpha': 255})

            self._create_particles(self.selected_cell, self.COLOR_SUCCESS, 30)
            self._create_particles((cx, cy), self.COLOR_SUCCESS, 30)

            self.grid[sy][sx] = None
            self.grid[cy][cx] = None
            self.pairs_left -= 1
            self.selected_cell = None
            self.selected_value = None
            return 10, "success"
        else:
            # --- Failure ---
            self._create_particles(self.selected_cell, self.COLOR_FAIL, 15)
            self._create_particles((cx, cy), self.COLOR_FAIL, 15)
            self.selected_cell = None
            self.selected_value = None
            return -1, "fail"
    
    def _calculate_movement_reward(self, old_pos, new_pos):
        """Calculates a small reward for moving towards a useful target."""
        if old_pos == new_pos:
            return 0

        old_dist = self._find_closest_target_dist(old_pos)
        new_dist = self._find_closest_target_dist(new_pos)

        if old_dist is not None and new_dist is not None:
            # Reward for reducing distance to a target
            return (old_dist - new_dist) * 0.1
        return 0

    def _find_closest_target_dist(self, from_pos):
        """Finds distance to the nearest target. Target depends on selection state."""
        min_dist = float('inf')
        target_found = False

        if self.selected_cell is None:
            # Target is any available number
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if self.grid[r][c] is not None and (c, r) != from_pos:
                        dist = abs(from_pos[0] - c) + abs(from_pos[1] - r)
                        if dist < min_dist:
                            min_dist = dist
                            target_found = True
        else:
            # Target is the matching number
            target_value = self.TARGET_SUM - self.selected_value
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if self.grid[r][c] == target_value:
                        dist = abs(from_pos[0] - c) + abs(from_pos[1] - r)
                        if dist < min_dist:
                            min_dist = dist
                            target_found = True
        
        return min_dist if target_found else None

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_grid()
        self._render_fading_cells()
        self._render_numbers()
        self._render_selection_and_cursor()
        self._update_and_render_particles()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.CELL_SIZE
            start_pos = (self.GRID_X_OFFSET, y)
            end_pos = (self.GRID_X_OFFSET + self.GRID_COLS * self.CELL_SIZE, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.CELL_SIZE
            start_pos = (x, self.GRID_Y_OFFSET)
            end_pos = (x, self.GRID_Y_OFFSET + self.GRID_ROWS * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start_pos, end_pos, 1)

    def _render_numbers(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                value = self.grid[r][c]
                if value is not None:
                    text_surf = self.number_font.render(str(value), True, self.COLOR_NUMBER)
                    text_rect = text_surf.get_rect(center=(
                        self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE // 2,
                        self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE // 2
                    ))
                    self.screen.blit(text_surf, text_rect)

    def _render_fading_cells(self):
        for cell in self.fading_cells[:]:
            cell['alpha'] -= 15
            if cell['alpha'] <= 0:
                self.fading_cells.remove(cell)
            else:
                c, r = cell['pos']
                color = (*self.COLOR_SUCCESS[:3], cell['alpha'])
                text_surf = self.number_font.render(str(cell['value']), True, self.COLOR_TEXT)
                text_surf.set_alpha(cell['alpha'])
                text_rect = text_surf.get_rect(center=(
                    self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE // 2,
                    self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE // 2
                ))
                self.screen.blit(text_surf, text_rect)

    def _render_selection_and_cursor(self):
        # Draw selection highlight
        if self.selected_cell is not None:
            sx, sy = self.selected_cell
            rect = pygame.Rect(
                self.GRID_X_OFFSET + sx * self.CELL_SIZE,
                self.GRID_Y_OFFSET + sy * self.CELL_SIZE,
                self.CELL_SIZE, self.CELL_SIZE
            )
            pygame.draw.rect(self.screen, self.COLOR_SELECTION, rect, 3, border_radius=5)

        # Draw cursor
        cx, cy = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_X_OFFSET + cx * self.CELL_SIZE,
            self.GRID_Y_OFFSET + cy * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        # Use gfxdraw for a smoother look
        pygame.gfxdraw.rectangle(self.screen, rect, (*self.COLOR_CURSOR, 150))
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, rect, 2, border_radius=5)

    def _create_particles(self, grid_pos, color, count):
        c, r = grid_pos
        center_x = self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(15, 30)
            self.particles.append({'pos': [center_x, center_y], 'vel': vel, 'life': lifetime, 'color': color})

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 30))
                pygame.gfxdraw.filled_circle(
                    self.screen, int(p['pos'][0]), int(p['pos'][1]), 2, (*p['color'], alpha)
                )

    def _render_ui(self):
        # Semi-transparent background for UI
        ui_surf = pygame.Surface((self.SCREEN_WIDTH, 35), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))

        # Score
        score_text = f"SCORE: {self.score}"
        score_surf = self.ui_font.render(score_text, True, self.COLOR_TEXT)
        self.screen.blit(score_surf, (10, 8))
        
        # Time
        time_text = f"TIME: {max(0, self.time_left):.1f}"
        time_surf = self.ui_font.render(time_text, True, self.COLOR_TEXT)
        time_rect = time_surf.get_rect(topright=(self.SCREEN_WIDTH - 10, 8))
        self.screen.blit(time_surf, time_rect)

        # Timer bar
        bar_width = self.SCREEN_WIDTH - 20
        time_ratio = max(0, self.time_left) / self.MAX_TIME
        current_width = int(bar_width * time_ratio)
        
        bar_color = self.COLOR_SUCCESS if time_ratio > 0.3 else self.COLOR_FAIL
        
        pygame.draw.rect(self.screen, self.COLOR_GRID, (10, self.SCREEN_HEIGHT - 15, bar_width, 5))
        if current_width > 0:
            pygame.draw.rect(self.screen, bar_color, (10, self.SCREEN_HEIGHT - 15, current_width, 5))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": self.time_left,
            "pairs_left": self.pairs_left,
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
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Number Matcher")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
            
        if keys[pygame.K_SPACE]:
            space_held = 1
            
        action = [movement, space_held, 0] # Shift is not used
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            running = False
            pygame.time.wait(2000) # Pause for 2 seconds before closing
            
        env.clock.tick(GameEnv.FPS)
        
    env.close()