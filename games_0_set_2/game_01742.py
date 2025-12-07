
# Generated: 2025-08-27T18:08:09.070711
# Source Brief: brief_01742.md
# Brief Index: 1742

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A fast-paced, grid-based tile-matching puzzle where strategic risk-taking
    leads to higher scores. The goal is to clear the board by matching groups
    of two or more colored tiles.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # User-facing strings
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press Space to match tiles."
    )
    game_description = (
        "A fast-paced, grid-based tile-matching puzzle. Match groups of same-colored tiles to clear the board and score points."
    )

    # Frame advance setting
    auto_advance = False

    # Game constants
    GRID_WIDTH = 12
    GRID_HEIGHT = 10
    TILE_SIZE = 32
    GRID_LINE_WIDTH = 2
    MAX_STEPS = 1000

    # Colors (Clean, vibrant, high-contrast)
    COLOR_BG = (44, 62, 80)          # #2c3e50 (Wet Asphalt)
    COLOR_GRID = (52, 73, 94)        # #34495e (Midnight Blue)
    COLOR_TEXT = (236, 240, 241)     # #ecf0f1 (Clouds)
    COLOR_SCORE_BAR = (40, 55, 71)
    COLOR_SELECTION = (255, 255, 255)

    TILE_COLORS = {
        1: (231, 76, 60),    # Red (#e74c3c Pomegranate)
        2: (46, 204, 113),   # Green (#2ecc71 Emerald)
        3: (52, 152, 219),   # Blue (#3498db Peter River)
        4: (149, 165, 166)   # Gray/Obstacle (#95a5a6 Concrete)
    }
    TILE_HIGHLIGHT_COLORS = {k: tuple(min(255, c + 40) for c in v) for k, v in TILE_COLORS.items()}

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = self.GRID_WIDTH * self.TILE_SIZE + self.GRID_LINE_WIDTH * (self.GRID_WIDTH + 1)
        self.screen_height = 400
        self.grid_offset_x = (640 - self.screen_width) // 2
        self.grid_offset_y = 60

        self.observation_space = Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        self.grid = None
        self.cursor_pos = None
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_gray_tiles = 5 # Initial difficulty

        self.np_random = None
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            # Fallback if no seed is provided
            if self.np_random is None:
                self.np_random = np.random.default_rng()


        self.steps = 0
        self.score = 0
        self.game_over = False

        self._generate_board()
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        return self._get_observation(), self._get_info()

    def _generate_board(self):
        while True:
            self.grid = self.np_random.integers(1, 4, size=(self.GRID_WIDTH, self.GRID_HEIGHT), endpoint=True)
            
            # Place obstacle tiles
            num_obstacles = min(self.level_gray_tiles, self.GRID_WIDTH * self.GRID_HEIGHT - 10) # Ensure some playable space
            obstacle_indices = self.np_random.choice(self.GRID_WIDTH * self.GRID_HEIGHT, num_obstacles, replace=False)
            for idx in obstacle_indices:
                x, y = idx % self.GRID_WIDTH, idx // self.GRID_WIDTH
                self.grid[x, y] = 4 # Gray tile
            
            if self._has_any_match():
                break

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_pressed, _ = action
        reward = 0

        # 1. Handle cursor movement
        if movement == 1:  # Up
            self.cursor_pos[1] = (self.cursor_pos[1] - 1 + self.GRID_HEIGHT) % self.GRID_HEIGHT
        elif movement == 2:  # Down
            self.cursor_pos[1] = (self.cursor_pos[1] + 1) % self.GRID_HEIGHT
        elif movement == 3:  # Left
            self.cursor_pos[0] = (self.cursor_pos[0] - 1 + self.GRID_WIDTH) % self.GRID_WIDTH
        elif movement == 4:  # Right
            self.cursor_pos[0] = (self.cursor_pos[0] + 1) % self.GRID_WIDTH

        # 2. Handle match attempt
        if space_pressed == 1:
            match_reward = self._attempt_match()
            reward += match_reward

        self.steps += 1
        
        # 3. Check for termination
        terminated = False
        if not self._has_any_match():
            if self._is_board_clear():
                # Win condition
                reward += 100 + 5 # Terminal reward + clear bonus
                self.level_gray_tiles += 1
            else:
                # Loss condition
                reward += -10
            terminated = True
        
        if self.steps >= self.MAX_STEPS:
            terminated = True

        self.game_over = terminated
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _attempt_match(self):
        x, y = self.cursor_pos
        tile_type = self.grid[x, y]

        if tile_type == 0 or tile_type == 4: # Empty or obstacle
            return -0.1

        connected_tiles = self._find_connected_tiles(x, y)
        
        if len(connected_tiles) >= 2:
            # Successful match
            # Sound: "match_success.wav"
            for cx, cy in connected_tiles:
                self.grid[cx, cy] = 0 # Mark as empty
            self._apply_gravity()
            return len(connected_tiles) # +1 point per tile
        else:
            # Failed match
            # Sound: "match_fail.wav"
            return -0.1

    def _find_connected_tiles(self, start_x, start_y):
        target_type = self.grid[start_x, start_y]
        if target_type == 0 or target_type == 4:
            return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        connected = []

        while q:
            x, y = q.popleft()
            connected.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT:
                    if (nx, ny) not in visited and self.grid[nx, ny] == target_type:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        
        return connected

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_slots = []
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] == 0:
                    empty_slots.append(y)
                elif empty_slots:
                    new_y = empty_slots.pop(0)
                    self.grid[x, new_y] = self.grid[x, y]
                    self.grid[x, y] = 0
                    empty_slots.append(y)
                    empty_slots.sort(reverse=True)

    def _has_any_match(self):
        visited = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in visited:
                    connected = self._find_connected_tiles(x, y)
                    if len(connected) >= 2:
                        return True
                    for pos in connected:
                        visited.add(pos)
        return False
    
    def _is_board_clear(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] in self.TILE_COLORS and self.grid[x,y] != 4:
                    return False
        return True

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        grid_pixel_width = self.GRID_WIDTH * self.TILE_SIZE + (self.GRID_WIDTH - 1) * self.GRID_LINE_WIDTH
        grid_pixel_height = self.GRID_HEIGHT * self.TILE_SIZE + (self.GRID_HEIGHT - 1) * self.GRID_LINE_WIDTH
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.grid_offset_x, self.grid_offset_y, grid_pixel_width, grid_pixel_height))

        # Draw tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                tile_type = self.grid[x, y]
                if tile_type != 0:
                    px = self.grid_offset_x + x * (self.TILE_SIZE + self.GRID_LINE_WIDTH)
                    py = self.grid_offset_y + y * (self.TILE_SIZE + self.GRID_LINE_WIDTH)
                    
                    color = self.TILE_COLORS[tile_type]
                    highlight_color = self.TILE_HIGHLIGHT_COLORS[tile_type]

                    # Main tile rect
                    pygame.draw.rect(self.screen, color, (px, py, self.TILE_SIZE, self.TILE_SIZE))
                    
                    # Inner highlight for 3D effect
                    pygame.draw.rect(self.screen, highlight_color, (px + 2, py + 2, self.TILE_SIZE - 4, self.TILE_SIZE - 4), border_radius=2)

        # Draw cursor with flashing effect
        cursor_x, cursor_y = self.cursor_pos
        px = self.grid_offset_x + cursor_x * (self.TILE_SIZE + self.GRID_LINE_WIDTH)
        py = self.grid_offset_y + cursor_y * (self.TILE_SIZE + self.GRID_LINE_WIDTH)
        
        # Flashing alpha
        alpha = int(128 + 127 * math.sin(self.steps * 0.3))
        
        # Use gfxdraw for anti-aliased, alpha-blended rectangle
        rect_surface = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
        pygame.gfxdraw.rectangle(rect_surface, (0, 0, self.TILE_SIZE - 1, self.TILE_SIZE - 1), (*self.COLOR_SELECTION, alpha))
        pygame.gfxdraw.rectangle(rect_surface, (1, 1, self.TILE_SIZE - 3, self.TILE_SIZE - 3), (*self.COLOR_SELECTION, alpha))
        self.screen.blit(rect_surface, (px, py))

    def _render_ui(self):
        # Top bar
        pygame.draw.rect(self.screen, self.COLOR_SCORE_BAR, (0, 0, 640, 50))
        pygame.draw.line(self.screen, self.COLOR_GRID, (0, 50), (640, 50), 2)
        
        # Score display
        score_text = self.font_large.render(f"Score: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (15, 12))
        
        # Steps/Time display
        steps_text = self.font_large.render(f"Steps: {self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        steps_rect = steps_text.get_rect(right=625, centery=25)
        self.screen.blit(steps_text, steps_rect)
        
        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((640, 400), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            status = "Board Cleared!" if self._is_board_clear() else "No More Moves"
            if self.steps >= self.MAX_STEPS and not self._is_board_clear():
                status = "Time Up!"

            game_over_text = self.font_large.render(status, True, self.COLOR_TEXT)
            text_rect = game_over_text.get_rect(center=(320, 200))
            overlay.blit(game_over_text, text_rect)
            self.screen.blit(overlay, (0,0))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
            "level_difficulty": self.level_gray_tiles,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example usage for testing
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    obs, info = env.reset()
    done = False
    
    # Create a window to display the game
    pygame.display.set_caption("Tile Matcher Gym Environment")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action[0] = 1
                elif event.key == pygame.K_DOWN:
                    action[0] = 2
                elif event.key == pygame.K_LEFT:
                    action[0] = 3
                elif event.key == pygame.K_RIGHT:
                    action[0] = 4
                elif event.key == pygame.K_SPACE:
                    action[1] = 1
                elif event.key == pygame.K_r: # Manual reset
                    obs, info = env.reset()
                    done = False
                elif event.key == pygame.K_ESCAPE:
                    running = False
        
        if not done:
            # Only step if an action was taken, since auto_advance is False
            if any(a != 0 for a in action):
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Done: {done}")
        
        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()