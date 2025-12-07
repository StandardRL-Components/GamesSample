
# Generated: 2025-08-27T17:00:58.142087
# Source Brief: brief_01400.md
# Brief Index: 1400

        
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
        "Controls: Use arrow keys to move the cursor. Press Space to clear the highlighted group of blocks. Press Shift to restart the game."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear the grid of colored blocks by matching groups of 3 or more in a limited number of moves. Plan your moves carefully to clear the entire board!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_ROWS, self.GRID_COLS = 10, 10
        self.GRID_OFFSET_X, self.GRID_OFFSET_Y = 120, 20
        self.CELL_SIZE = 36
        self.MAX_MOVES = 10
        self.MAX_EPISODE_STEPS = 1000

        self.COLOR_BG = (25, 30, 35)
        self.COLOR_GRID = (50, 60, 70)
        self.BLOCK_COLORS = [
            (255, 80, 80),    # Red
            (80, 255, 80),    # Green
            (80, 150, 255),   # Blue
            (255, 255, 80),   # Yellow
            (200, 80, 255),   # Purple
        ]
        self.COLOR_CURSOR = (255, 255, 255, 150)
        self.COLOR_HIGHLIGHT = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SCORE = (255, 200, 0)
        self.COLOR_OVERLAY = (0, 0, 0, 180)

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State ---
        # These are initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.moves_remaining = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.particles = None
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.moves_remaining = self.MAX_MOVES
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.cursor_pos = np.array([self.GRID_COLS // 2, self.GRID_ROWS // 2])
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_grid()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        terminated = False
        
        # Handle restart action (rising edge of shift)
        if shift_held and not self.prev_shift_held:
            obs, info = self.reset()
            return obs, -1.0, False, False, info # Small penalty for restarting

        self.prev_shift_held = shift_held

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # 1. Update Game Logic
        self._update_cursor(movement)
        
        # Handle select action (rising edge of space)
        if space_held and not self.prev_space_held:
            reward += self._attempt_clear_group()
        
        self.prev_space_held = space_held

        self._update_particles()

        # 2. Check for Termination
        if self._is_grid_clear():
            self.game_over = True
            terminated = True
            reward += 100 # Win bonus
        elif self.moves_remaining <= 0:
            self.game_over = True
            terminated = True
            reward += -100 # Loss penalty
            
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True

        self.steps += 1
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info(),
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_remaining": self.moves_remaining}

    # --- Helper Functions: Game Logic ---

    def _generate_grid(self):
        while True:
            self.grid = self.np_random.integers(0, len(self.BLOCK_COLORS), size=(self.GRID_ROWS, self.GRID_COLS))
            if self._has_valid_move():
                break

    def _has_valid_move(self):
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.grid[r, c] != -1:
                    group = self._find_connected_group(c, r)
                    if len(group) >= 3:
                        return True
        return False
        
    def _is_grid_clear(self):
        return np.all(self.grid == -1)

    def _update_cursor(self, movement):
        if movement == 1: self.cursor_pos[1] -= 1  # Up
        elif movement == 2: self.cursor_pos[1] += 1  # Down
        elif movement == 3: self.cursor_pos[0] -= 1  # Left
        elif movement == 4: self.cursor_pos[0] += 1  # Right
        self.cursor_pos[0] = np.clip(self.cursor_pos[0], 0, self.GRID_COLS - 1)
        self.cursor_pos[1] = np.clip(self.cursor_pos[1], 0, self.GRID_ROWS - 1)

    def _attempt_clear_group(self):
        cx, cy = self.cursor_pos
        if self.grid[cy, cx] == -1:
            return -0.1 # Penalty for clicking empty space

        group = self._find_connected_group(cx, cy)
        
        if len(group) >= 3:
            # // Sound: Block clear
            self.moves_remaining -= 1
            
            # Remove blocks and create particles
            color_index = self.grid[group[0][1], group[0][0]]
            for x, y in group:
                self.grid[y, x] = -1
                self._spawn_particles(x, y, color_index)
            
            self._apply_gravity_and_refill()
            
            # Calculate reward
            reward = len(group) # +1 per block
            if len(group) >= 4:
                reward += 5 # Bonus
            self.score += reward
            return reward
        else:
            # // Sound: Invalid move
            self.moves_remaining -= 1 # Penalty for invalid attempt
            self.score -= 0.1
            return -0.1

    def _find_connected_group(self, start_x, start_y):
        if self.grid[start_y, start_x] == -1:
            return []

        target_color = self.grid[start_y, start_x]
        q = [(start_x, start_y)]
        visited = set(q)
        group = []

        while q:
            x, y = q.pop(0)
            group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS:
                    if (nx, ny) not in visited and self.grid[ny, nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            empty_row = self.GRID_ROWS - 1
            for r in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[r, c] != -1:
                    self.grid[empty_row, c], self.grid[r, c] = self.grid[r, c], self.grid[empty_row, c]
                    empty_row -= 1
        
        # Refill
        self.grid[self.grid == -1] = self.np_random.integers(0, len(self.BLOCK_COLORS), size=np.count_nonzero(self.grid == -1))


    def _spawn_particles(self, grid_x, grid_y, color_index):
        px = self.GRID_OFFSET_X + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_OFFSET_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.BLOCK_COLORS[color_index]
        
        for _ in range(10): # Spawn 10 particles per block
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = self.np_random.integers(20, 40)
            self.particles.append([px, py, vx, vy, life, color])

    def _update_particles(self):
        for p in self.particles:
            p[0] += p[1] # x += vx
            p[1] += p[2] # y += vy
            p[3] -= 1    # life -= 1
        self.particles = [p for p in self.particles if p[3] > 0]


    # --- Helper Functions: Rendering ---

    def _render_game(self):
        # Draw grid lines
        grid_width = self.GRID_COLS * self.CELL_SIZE
        grid_height = self.GRID_ROWS * self.CELL_SIZE
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_OFFSET_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_OFFSET_Y), (x, self.GRID_OFFSET_Y + grid_height))
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_OFFSET_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, y), (self.GRID_OFFSET_X + grid_width, y))

        # Draw blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = self.grid[r, c]
                if color_index != -1:
                    color = self.BLOCK_COLORS[color_index]
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + c * self.CELL_SIZE + 2,
                        self.GRID_OFFSET_Y + r * self.CELL_SIZE + 2,
                        self.CELL_SIZE - 4,
                        self.CELL_SIZE - 4
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)

        # Draw highlighted group
        if not self.game_over:
            highlight_group = self._find_connected_group(self.cursor_pos[0], self.cursor_pos[1])
            if len(highlight_group) >= 3:
                for c, r in highlight_group:
                    rect = pygame.Rect(
                        self.GRID_OFFSET_X + c * self.CELL_SIZE,
                        self.GRID_OFFSET_Y + r * self.CELL_SIZE,
                        self.CELL_SIZE,
                        self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, self.COLOR_HIGHLIGHT, rect, width=2, border_radius=6)

        # Draw cursor
        if not self.game_over:
            cursor_rect = pygame.Rect(
                self.GRID_OFFSET_X + self.cursor_pos[0] * self.CELL_SIZE,
                self.GRID_OFFSET_Y + self.cursor_pos[1] * self.CELL_SIZE,
                self.CELL_SIZE,
                self.CELL_SIZE
            )
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(self.COLOR_CURSOR)
            self.screen.blit(s, cursor_rect.topleft)

        # Draw particles
        for p in self.particles:
            pos = (int(p[0]), int(p[1]))
            life_ratio = max(0, p[3] / 40.0)
            size = int(life_ratio * 5)
            if size > 0:
                color = tuple(int(c * life_ratio) for c in p[5])
                pygame.draw.rect(self.screen, color, (pos[0] - size//2, pos[1] - size//2, size, size))

    def _render_ui(self):
        # UI Panel on the right
        ui_x = self.GRID_OFFSET_X + self.GRID_COLS * self.CELL_SIZE + 30
        
        # Score
        score_text = self.font_small.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (ui_x, 50))
        score_val = self.font_main.render(f"{self.score:.1f}", True, self.COLOR_SCORE)
        self.screen.blit(score_val, (ui_x, 75))
        
        # Moves
        moves_text = self.font_small.render("MOVES LEFT", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (ui_x, 150))
        moves_val = self.font_main.render(str(self.moves_remaining), True, self.COLOR_TEXT)
        self.screen.blit(moves_val, (ui_x, 175))

        # Game Over Screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_OVERLAY)
            self.screen.blit(overlay, (0, 0))
            
            if self._is_grid_clear():
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)
                
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)
            
            restart_text = self.font_small.render("Press Shift to play again", True, self.COLOR_TEXT)
            restart_rect = restart_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2 + 50))
            self.screen.blit(restart_text, restart_rect)


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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This block will not be executed by the grading system but is useful for testing.
    # Set `render_mode="human"` to see the game window.
    env = GameEnv()
    
    # --- Manual Play ---
    # To play manually, you need to run this in a non-headless environment
    # and create a Pygame window to display the rendered frames.
    try:
        import os
        os.environ["SDL_VIDEODRIVER"] = "x11" # Or "windows", "macOS"
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Block Clear")
    except pygame.error:
        print("Could not set up a display for manual play. Pygame might be in headless mode.")
        screen = None

    if screen:
        obs, info = env.reset()
        done = False
        clock = pygame.time.Clock()

        while not done:
            # Action mapping for human keyboard
            keys = pygame.key.get_pressed()
            movement = 0 # none
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # For turn-based games, only step on an event
            event_occurred = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    event_occurred = True

            if event_occurred:
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    print(f"Game Over. Score: {info['score']}")
                    # In a real scenario, you might wait for a restart key
                    # For this simple loop, we'll just exit.
                    # done = True 
            
            # Render the observation to the screen
            frame = np.transpose(obs, (1, 0, 2))
            surf = pygame.surfarray.make_surface(frame)
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit frame rate

    env.close()