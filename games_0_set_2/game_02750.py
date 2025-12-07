
# Generated: 2025-08-27T21:19:40.421512
# Source Brief: brief_02750.md
# Brief Index: 2750

        
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
        "Controls: Arrow keys to move the selector. Press space to match fruits."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match cascading fruits in a grid-based puzzle to reach a target score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 8
    GRID_ROWS = 8
    CELL_SIZE = 40
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_COLS * CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_ROWS * CELL_SIZE) // 2 + 20

    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    WIN_SCORE = 5000
    MIN_MATCH_COUNT = 3
    
    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (30, 40, 50)
    COLOR_GRID_LINE = (50, 60, 70)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    
    FRUIT_COLORS = {
        1: (255, 50, 50),   # Red: Apple
        2: (255, 255, 50),  # Yellow: Banana
        3: (50, 255, 50),   # Green: Lime
        4: (50, 150, 255),  # Blue: Blueberry
        5: (255, 150, 50),  # Orange: Orange
        6: (200, 50, 255),  # Purple: Grape
    }
    NUM_FRUIT_TYPES = len(FRUIT_COLORS)

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
        self.font_large = pygame.font.SysFont("monospace", 36, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 18)

        # Game state variables are initialized in reset()
        self.grid = None
        self.visual_grid_y = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.time_remaining = None
        self.particles = None
        self.last_space_held = None
        self.move_cooldown = 0

        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.time_remaining = self.MAX_STEPS
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.particles = []
        self.last_space_held = False
        self.move_cooldown = 0
        
        self._init_grid()
        
        return self._get_observation(), self._get_info()

    def _init_grid(self):
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.visual_grid_y = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=float)

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Place initial fruits far above the screen to animate them falling in
                self.visual_grid_y[r, c] = (r - self.GRID_ROWS) * self.CELL_SIZE
        
        # Fill grid with random fruits, ensuring no initial matches
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                possible_fruits = list(self.FRUIT_COLORS.keys())
                # Check left and above neighbors to avoid creating a match
                if c >= 2 and self.grid[r, c-1] == self.grid[r, c-2]:
                    if self.grid[r, c-1] in possible_fruits:
                        possible_fruits.remove(self.grid[r, c-1])
                if r >= 2 and self.grid[r-1, c] == self.grid[r-2, c]:
                    if self.grid[r-1, c] in possible_fruits:
                        possible_fruits.remove(self.grid[r-1, c])
                
                self.grid[r, c] = self.np_random.choice(possible_fruits) if possible_fruits else 1
    
    def step(self, action):
        reward = 0
        self.steps += 1
        self.time_remaining -= 1
        self.move_cooldown = max(0, self.move_cooldown - 1)

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Handle movement with a cooldown to prevent frantic cursor
        if self.move_cooldown == 0:
            moved = False
            if movement == 1 and self.cursor_pos[1] > 0:  # Up
                self.cursor_pos[1] -= 1
                moved = True
            elif movement == 2 and self.cursor_pos[1] < self.GRID_ROWS - 1:  # Down
                self.cursor_pos[1] += 1
                moved = True
            elif movement == 3 and self.cursor_pos[0] > 0:  # Left
                self.cursor_pos[0] -= 1
                moved = True
            elif movement == 4 and self.cursor_pos[0] < self.GRID_COLS - 1:  # Right
                self.cursor_pos[0] += 1
                moved = True
            if moved:
                self.move_cooldown = 3 # frames cooldown

        # Handle match on space press (not hold)
        if space_held and not self.last_space_held:
            c, r = self.cursor_pos
            if self.grid[r, c] != 0:
                matches = self._find_matches(r, c)
                if len(matches) >= self.MIN_MATCH_COUNT:
                    # sfx: match_sound()
                    match_reward = len(matches)
                    self.score += match_reward * 10
                    reward += match_reward

                    if len(matches) > 4:
                        reward += 10
                        self.score += 100 # Combo bonus
                    
                    for r_m, c_m in matches:
                        self._create_particles(c_m, r_m, self.grid[r_m, c_m])
                        self.grid[r_m, c_m] = 0 # Mark as empty
                    
                    self._cascade_and_refill()

        self.last_space_held = space_held

        # --- Update Game State ---
        self._update_particles()
        self._update_visuals()

        # --- Termination Check ---
        terminated = False
        if self.score >= self.WIN_SCORE:
            reward += 100
            terminated = True
            self.game_over = True
        elif self.time_remaining <= 0:
            reward -= 10
            terminated = True
            self.game_over = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _find_matches(self, start_r, start_c):
        target_fruit = self.grid[start_r, start_c]
        if target_fruit == 0:
            return []
        
        q = [(start_r, start_c)]
        visited = set(q)
        matches = []

        while q:
            r, c = q.pop(0)
            matches.append((r, c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS:
                    if (nr, nc) not in visited and self.grid[nr, nc] == target_fruit:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        return matches

    def _cascade_and_refill(self):
        # Let fruits fall down
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != 0:
                    if r != empty_row:
                        self.grid[empty_row, c] = self.grid[r, c]
                        self.visual_grid_y[empty_row, c] = self.visual_grid_y[r, c]
                        self.grid[r, c] = 0
                    empty_row -= 1
        
        # Refill top rows with new fruits
        for c in range(self.GRID_COLS):
            for r in range(self.GRID_ROWS):
                if self.grid[r, c] == 0:
                    self.grid[r, c] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)
                    # Place new fruits above the screen to fall in
                    self.visual_grid_y[r, c] = (r - self.GRID_ROWS) * self.CELL_SIZE

    def _update_visuals(self):
        # Animate fruits falling into place with easing
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                target_y = r * self.CELL_SIZE
                current_y = self.visual_grid_y[r, c]
                self.visual_grid_y[r, c] += (target_y - current_y) * 0.2

    def _create_particles(self, c, r, fruit_type):
        color = self.FRUIT_COLORS.get(fruit_type, (255, 255, 255))
        center_x = self.GRID_X_OFFSET + c * self.CELL_SIZE + self.CELL_SIZE / 2
        center_y = self.GRID_Y_OFFSET + r * self.CELL_SIZE + self.CELL_SIZE / 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(2, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = self.np_random.integers(15, 30)
            size = self.np_random.uniform(2, 5)
            self.particles.append([center_x, center_y, vx, vy, lifetime, color, size])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]  # x += vx
            p[1] += p[3]  # y += vy
            p[4] -= 1     # lifetime -= 1
            p[2] *= 0.98  # drag
            p[3] *= 0.98  # drag

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_COLS * self.CELL_SIZE, self.GRID_ROWS * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # Draw fruits
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                fruit_type = self.grid[r, c]
                if fruit_type != 0:
                    visual_y = self.visual_grid_y[r, c]
                    x_pos = self.GRID_X_OFFSET + c * self.CELL_SIZE
                    y_pos = self.GRID_Y_OFFSET + visual_y
                    
                    # Only draw if visible on screen
                    if y_pos > self.GRID_Y_OFFSET - self.CELL_SIZE:
                        self._render_fruit(x_pos, y_pos, fruit_type)

        # Draw grid lines on top
        for r in range(self.GRID_ROWS + 1):
            start = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + r * self.CELL_SIZE)
            end = (self.GRID_X_OFFSET + self.GRID_COLS * self.CELL_SIZE, self.GRID_Y_OFFSET + r * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start, end, 1)
        for c in range(self.GRID_COLS + 1):
            start = (self.GRID_X_OFFSET + c * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end = (self.GRID_X_OFFSET + c * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_ROWS * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, start, end, 1)

        self._render_cursor()
        self._render_particles()

    def _render_fruit(self, x, y, fruit_type):
        color = self.FRUIT_COLORS.get(fruit_type, (0,0,0))
        padding = 4
        fruit_rect = pygame.Rect(x + padding, y + padding, self.CELL_SIZE - padding * 2, self.CELL_SIZE - padding * 2)
        
        # Main fruit body
        pygame.draw.rect(self.screen, color, fruit_rect, border_radius=8)
        
        # Shine effect
        shine_color = tuple(min(255, c + 60) for c in color)
        shine_center = (int(x + self.CELL_SIZE * 0.35), int(y + self.CELL_SIZE * 0.35))
        pygame.draw.circle(self.screen, shine_color, shine_center, int(self.CELL_SIZE * 0.15))

    def _render_cursor(self):
        c, r = self.cursor_pos
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + c * self.CELL_SIZE,
            self.GRID_Y_OFFSET + r * self.CELL_SIZE,
            self.CELL_SIZE,
            self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=4)

    def _render_particles(self):
        for x, y, vx, vy, lifetime, color, size in self.particles:
            alpha = max(0, min(255, int(255 * (lifetime / 30.0))))
            p_color = (color[0], color[1], color[2], alpha)
            
            # Use gfxdraw for alpha blending
            s = pygame.Surface((int(size*2), int(size*2)), pygame.SRCALPHA)
            pygame.draw.rect(s, p_color, s.get_rect())
            self.screen.blit(s, (int(x - size), int(y - size)))

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 5))

        # Time display
        time_str = f"TIME: {self.time_remaining // self.FPS:02d}"
        time_text = self.font_large.render(time_str, True, self.COLOR_TEXT)
        time_rect = time_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 5))
        self.screen.blit(time_text, time_rect)

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            status_text_str = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME'S UP!"
            status_text = self.font_large.render(status_text_str, True, (255, 255, 100))
            status_rect = status_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 - 20))
            
            score_text = self.font_small.render(f"Final Score: {self.score}", True, self.COLOR_TEXT)
            score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 20))
            
            overlay.blit(status_text, status_rect)
            overlay.blit(score_text, score_rect)
            self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_remaining_seconds": self.time_remaining // self.FPS,
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # Set this to "human" to see the game being played
    render_mode = "human" # "rgb_array" or "human"
    
    if render_mode == "human":
        # Override the screen for human rendering
        class HumanGameEnv(GameEnv):
            def __init__(self, render_mode="human"):
                super().__init__(render_mode)
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
                pygame.display.set_caption("Fruit Matcher")
            
            def _get_observation(self):
                # In human mode, we render to the display screen
                super()._get_observation() # This draws on the internal surface
                self.screen.blit(super().screen, (0,0))
                pygame.display.flip()
                # We still need to return the array for compatibility, 
                # even if it's not the primary display.
                arr = pygame.surfarray.array3d(super().screen)
                return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
        
        env = HumanGameEnv()
    else:
        env = GameEnv()

    obs, info = env.reset()
    done = False
    
    # Key mapping for human play
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }

    # Main game loop for human player
    while not done:
        # Default action is no-op
        movement = 0
        space = 0
        shift = 0

        if render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            keys = pygame.key.get_pressed()
            for key, move_action in key_map.items():
                if keys[key]:
                    movement = move_action
                    break # Prioritize one movement key
            
            if keys[pygame.K_SPACE]:
                space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift = 1
            
            action = [movement, space, shift]
        else:
            # For testing, use random actions
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        
        if render_mode == "human":
            env.clock.tick(env.FPS)
        
        if terminated or truncated:
            print(f"Game Over! Final Info: {info}")
            # In a real scenario, you might reset here. For a single run, we just exit.
            done = True

    if render_mode == "human":
        pygame.time.wait(2000) # Wait 2 seconds before closing
    env.close()