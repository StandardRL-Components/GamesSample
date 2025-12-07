
# Generated: 2025-08-28T07:12:37.102710
# Source Brief: brief_03179.md
# Brief Index: 3179

        
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
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Arrow keys to move cursor. Hold Shift to preview a match. Press Space to clear matching tiles."
    )

    game_description = (
        "Match 3 or more adjacent colored tiles to score points. Reach 100 points within 20 moves to win!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 8
        self.TILE_SIZE = 40
        self.GRID_MARGIN_X = (self.SCREEN_WIDTH - (self.GRID_WIDTH * self.TILE_SIZE)) // 2
        self.GRID_MARGIN_Y = (self.SCREEN_HEIGHT - (self.GRID_HEIGHT * self.TILE_SIZE)) // 2
        
        self.INITIAL_MOVES = 20
        self.WIN_SCORE = 100
        self.MAX_STEPS = 1000

        # --- Rewards ---
        self.REWARD_WIN = 100.0
        self.REWARD_LOSE = -50.0
        self.REWARD_INVALID_MOVE = -0.1

        # --- Colors ---
        self.COLORS = [
            pygame.Color("#ff6b6b"), # Red
            pygame.Color("#48dbfb"), # Blue
            pygame.Color("#1dd1a1"), # Green
            pygame.Color("#feca57"), # Yellow
            pygame.Color("#ff9ff3"), # Pink
            pygame.Color("#5f27cd"), # Purple
        ]
        self.COLOR_BG = pygame.Color("#1e1e1e")
        self.COLOR_GRID = pygame.Color("#333333")
        self.COLOR_CURSOR = pygame.Color("#ffffff")
        self.COLOR_UI_TEXT = pygame.Color("#dddddd")

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_game_over = pygame.font.SysFont("Consolas", 50, bold=True)

        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.steps = None
        self.game_over = None
        self.last_space_held = None
        self.preview_group = None
        self.particles = None
        self.np_random = None

        self.reset()
        
        # self.validate_implementation() # Optional: Call for self-testing

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            # Fallback to a default or existing generator
            if self.np_random is None:
                self.np_random = np.random.default_rng()

        self.score = 0
        self.moves_left = self.INITIAL_MOVES
        self.steps = 0
        self.game_over = False
        self.last_space_held = False
        self.preview_group = []
        self.particles = []
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Ensure the starting grid has at least one valid move
        while True:
            self.grid = self.np_random.integers(0, len(self.COLORS), size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if self._has_valid_moves():
                break

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Clear one-frame effects
        self.particles.clear()
        self.preview_group.clear()

        reward = 0
        
        # --- Handle Input ---
        self._handle_movement(movement)

        if shift_held:
            self.preview_group = self._find_connected_tiles(self.cursor_pos[0], self.cursor_pos[1])
            if len(self.preview_group) < 3:
                self.preview_group = [] # Don't preview invalid moves

        space_pressed = space_held and not self.last_space_held
        if space_pressed:
            reward += self._process_match_attempt()
        
        self.last_space_held = space_held
        
        # --- Update Game State ---
        self.steps += 1
        terminated = self._check_termination()
        
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += self.REWARD_WIN
            else:
                reward += self.REWARD_LOSE

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1: # Up
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: # Down
            self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: # Left
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: # Right
            self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

    def _process_match_attempt(self):
        match_group = self._find_connected_tiles(self.cursor_pos[0], self.cursor_pos[1])

        if len(match_group) < 3:
            # Sound: invalid move sfx
            return self.REWARD_INVALID_MOVE

        # --- Successful Match ---
        # Sound: match success sfx
        self.moves_left -= 1
        
        score_increase = (len(match_group) - 2) ** 2
        self.score += score_increase
        
        match_color = self.COLORS[self.grid[self.cursor_pos[1]][self.cursor_pos[0]]]
        
        for x, y in match_group:
            self.grid[y][x] = -1  # Mark as empty
            self._create_particles((x, y), match_color)

        self._handle_tile_fall()
        
        return float(score_increase)

    def _check_termination(self):
        if self.score >= self.WIN_SCORE:
            return True
        if self.moves_left <= 0:
            return True
        if not self._has_valid_moves():
            return True
        if self.steps >= self.MAX_STEPS:
            return True
        return False
        
    def _find_connected_tiles(self, start_x, start_y):
        if not (0 <= start_x < self.GRID_WIDTH and 0 <= start_y < self.GRID_HEIGHT):
            return []

        target_color_idx = self.grid[start_y][start_x]
        if target_color_idx == -1: return []

        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        group = []

        while q:
            x, y = q.popleft()
            group.append((x, y))

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    if self.grid[ny][nx] == target_color_idx:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return group

    def _has_valid_moves(self):
        visited = np.zeros_like(self.grid, dtype=bool)
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if not visited[y][x]:
                    group = self._find_connected_tiles(x, y)
                    if len(group) >= 3:
                        return True
                    for gx, gy in group:
                        visited[gy][gx] = True
        return False

    def _handle_tile_fall(self):
        # Sound: tiles falling sfx
        for x in range(self.GRID_WIDTH):
            write_idx = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] != -1:
                    if y != write_idx:
                        self.grid[write_idx][x] = self.grid[y][x]
                        self.grid[y][x] = -1
                    write_idx -= 1
            
            # Fill empty spaces at the top
            for y in range(write_idx, -1, -1):
                self.grid[y][x] = self.np_random.integers(0, len(self.COLORS))

    def _create_particles(self, pos, color):
        tile_center_x = self.GRID_MARGIN_X + pos[0] * self.TILE_SIZE + self.TILE_SIZE // 2
        tile_center_y = self.GRID_MARGIN_Y + pos[1] * self.TILE_SIZE + self.TILE_SIZE // 2
        for _ in range(10): # 10 particles per tile
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            p = {
                "x": tile_center_x,
                "y": tile_center_y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed,
                "life": self.np_random.uniform(0.5, 1.0), # Will be one-frame anyway
                "color": color,
                "size": self.np_random.integers(2, 5)
            }
            self.particles.append(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_MARGIN_X + x * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_MARGIN_Y), (px, self.GRID_MARGIN_Y + self.GRID_HEIGHT * self.TILE_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_MARGIN_Y + y * self.TILE_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_MARGIN_X, py), (self.GRID_MARGIN_X + self.GRID_WIDTH * self.TILE_SIZE, py))

        # Draw tiles
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y][x]
                if color_idx != -1:
                    color = self.COLORS[color_idx]
                    rect = pygame.Rect(
                        self.GRID_MARGIN_X + x * self.TILE_SIZE + 2,
                        self.GRID_MARGIN_Y + y * self.TILE_SIZE + 2,
                        self.TILE_SIZE - 4,
                        self.TILE_SIZE - 4
                    )
                    pygame.draw.rect(self.screen, color, rect, border_radius=4)
        
        # Draw preview selection (shift held)
        if self.preview_group:
            for x, y in self.preview_group:
                color_idx = self.grid[y][x]
                if color_idx != -1:
                    preview_color = self.COLORS[color_idx].lerp((255,255,255), 0.5)
                    rect = pygame.Rect(
                        self.GRID_MARGIN_X + x * self.TILE_SIZE,
                        self.GRID_MARGIN_Y + y * self.TILE_SIZE,
                        self.TILE_SIZE,
                        self.TILE_SIZE
                    )
                    s = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                    s.fill((*preview_color[:3], 100))
                    self.screen.blit(s, rect.topleft)
        
        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_MARGIN_X + self.cursor_pos[0] * self.TILE_SIZE,
            self.GRID_MARGIN_Y + self.cursor_pos[1] * self.TILE_SIZE,
            self.TILE_SIZE,
            self.TILE_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, 3, border_radius=6)

        # Draw particles
        for p in self.particles:
            pygame.draw.rect(self.screen, p["color"], (int(p["x"]), int(p["y"]), p["size"], p["size"]))

    def _render_ui(self):
        # Score display
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Moves display
        moves_text = self.font_ui.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 10, 10))

        # Game Over display
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))

            if self.score >= self.WIN_SCORE:
                end_text = self.font_game_over.render("YOU WIN!", True, self.COLORS[3])
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLORS[0])
            
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "steps": self.steps,
            "cursor_pos": list(self.cursor_pos),
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
        print("--- Running Implementation Validation ---")
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


if __name__ == "__main__":
    # --- Manual Play Example ---
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Tile Matcher")
    clock = pygame.time.Clock()

    running = True
    terminated = False
    
    while running:
        # --- Action Mapping for Manual Play ---
        movement = 0 # no-op
        space = 0
        shift = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'R' key
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']}, Moves: {info['moves_left']}")

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        clock.tick(30) # Limit frame rate for manual play

    env.close()