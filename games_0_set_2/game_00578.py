
# Generated: 2025-08-27T14:05:27.053996
# Source Brief: brief_00578.md
# Brief Index: 578

        
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
        "Controls: Use arrow keys to move the cursor. Press space to select a gem. "
        "Use arrow keys again to swap with an adjacent gem. Press shift to deselect."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Strategically match gems in a grid to reach a target score before running out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    # --- Constants ---
    GRID_WIDTH = 8
    GRID_HEIGHT = 8
    NUM_GEM_TYPES = 6
    TARGET_SCORE = 500
    MAX_MOVES = 20
    
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Colors
    COLOR_BG = (20, 25, 40)
    COLOR_GRID = (40, 50, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_CURSOR = (255, 255, 255, 100)
    COLOR_SELECT = (255, 255, 0)
    COLOR_MATCH_FLASH = (255, 255, 255, 200)

    GEM_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (255, 80, 255),  # Magenta
        (80, 255, 255),  # Cyan
    ]

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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_desc = pygame.font.SysFont("Consolas", 18)

        # Calculate grid rendering properties
        self.grid_rect = pygame.Rect(
            (self.SCREEN_WIDTH - self.SCREEN_HEIGHT) // 2, 0,
            self.SCREEN_HEIGHT, self.SCREEN_HEIGHT
        )
        self.cell_size = self.grid_rect.width // self.GRID_WIDTH
        
        # Initialize state variables
        self.grid = None
        self.np_random = None
        self.steps = 0
        self.score = 0
        self.moves_left = 0
        self.game_over = False
        self.cursor_pos = (0, 0)
        self.selected_gem_pos = None
        self.last_matched_gems = set()
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.moves_left = self.MAX_MOVES
        self.game_over = False
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.selected_gem_pos = None
        self.last_matched_gems = set()
        
        self._initialize_board()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1
        
        reward = 0
        self.last_matched_gems = set() # Clear flash effect from previous step
        self.steps += 1
        
        # Action handling priority: Deselect > Swap > Select > Move
        if shift_held and self.selected_gem_pos is not None:
            self.selected_gem_pos = None
            # sound: "deselect"
        elif self.selected_gem_pos is not None and movement != 0:
            reward = self._handle_swap_attempt(movement)
        elif space_held:
            if self.selected_gem_pos is None:
                self.selected_gem_pos = self.cursor_pos
                # sound: "select"
        elif movement != 0:
            if self.selected_gem_pos is None:
                self._move_cursor(movement)
                # sound: "cursor_move"

        terminated = self._check_termination()
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.TARGET_SCORE:
                reward += 100 # Win bonus
                # sound: "win"
            else:
                reward += -50 # Lose penalty
                # sound: "lose"

        # Anti-softlock: reshuffle if no moves are possible
        if not self.game_over and not self._has_possible_moves():
            self._shuffle_board()
            # sound: "shuffle"

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,  # truncated always False
            self._get_info()
        )
    
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
            "moves_left": self.moves_left,
            "cursor_pos": self.cursor_pos,
            "selected": self.selected_gem_pos is not None,
        }

    # --- Game Logic Helpers ---
    
    def _initialize_board(self):
        while True:
            self.grid = self.np_random.integers(0, self.NUM_GEM_TYPES, size=(self.GRID_HEIGHT, self.GRID_WIDTH))
            if not self._find_all_matches(self.grid) and self._has_possible_moves():
                break

    def _shuffle_board(self):
        flat_grid = self.grid.flatten().tolist()
        self.np_random.shuffle(flat_grid)
        self.grid = np.array(flat_grid).reshape((self.GRID_HEIGHT, self.GRID_WIDTH))
        
        if self._find_all_matches(self.grid) or not self._has_possible_moves():
            self._initialize_board()

    def _move_cursor(self, movement):
        x, y = self.cursor_pos
        if movement == 1: y -= 1  # Up
        elif movement == 2: y += 1  # Down
        elif movement == 3: x -= 1  # Left
        elif movement == 4: x += 1  # Right
        self.cursor_pos = (max(0, min(self.GRID_WIDTH - 1, x)), max(0, min(self.GRID_HEIGHT - 1, y)))

    def _handle_swap_attempt(self, movement):
        x1, y1 = self.selected_gem_pos
        x2, y2 = x1, y1

        if movement == 1: y2 -= 1
        elif movement == 2: y2 += 1
        elif movement == 3: x2 -= 1
        elif movement == 4: x2 += 1

        if not (0 <= x2 < self.GRID_WIDTH and 0 <= y2 < self.GRID_HEIGHT):
            self.selected_gem_pos = None
            return -0.1

        self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
        
        cleared_gems, score_gain = self._process_matches()
        
        if cleared_gems > 0:
            self.moves_left -= 1
            self.score += score_gain
            # sound: "match_success"
            reward = cleared_gems * 1.0
            if cleared_gems == 4: reward += 10
            elif cleared_gems >= 5: reward += 20
        else:
            self.grid[y1, x1], self.grid[y2, x2] = self.grid[y2, x2], self.grid[y1, x1]
            # sound: "match_fail"
            reward = -0.1

        self.selected_gem_pos = None
        return reward

    def _process_matches(self):
        total_cleared = 0
        total_score = 0
        
        while True:
            matches = self._find_all_matches(self.grid)
            if not matches: break
                
            num_cleared = len(matches)
            total_cleared += num_cleared
            score_gain = num_cleared * 10
            if num_cleared == 4: score_gain += 20
            elif num_cleared >= 5: score_gain += 40
            total_score += score_gain

            self.last_matched_gems.update(matches)
            
            for y, x in matches: self.grid[y, x] = -1
            
            self._collapse_grid()
            self._refill_grid()
            
        return total_cleared, total_score

    def _find_all_matches(self, grid):
        matches = set()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH - 2):
                if grid[r, c] != -1 and grid[r, c] == grid[r, c+1] == grid[r, c+2]:
                    gem_type = grid[r, c]
                    i = c
                    while i < self.GRID_WIDTH and grid[r, i] == gem_type: matches.add((r, i)); i += 1
        for c in range(self.GRID_WIDTH):
            for r in range(self.GRID_HEIGHT - 2):
                if grid[r, c] != -1 and grid[r, c] == grid[r+1, c] == grid[r+2, c]:
                    gem_type = grid[r, c]
                    i = r
                    while i < self.GRID_HEIGHT and grid[i, c] == gem_type: matches.add((i, c)); i += 1
        return matches

    def _collapse_grid(self):
        for c in range(self.GRID_WIDTH):
            write_idx = self.GRID_HEIGHT - 1
            for r in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[r, c] != -1:
                    if r != write_idx: self.grid[write_idx, c] = self.grid[r, c]
                    write_idx -= 1
            for r in range(write_idx, -1, -1): self.grid[r, c] = -1

    def _refill_grid(self):
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] == -1:
                    self.grid[r, c] = self.np_random.integers(0, self.NUM_GEM_TYPES)

    def _has_possible_moves(self):
        temp_grid = self.grid.copy()
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if c < self.GRID_WIDTH - 1:
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                    if self._find_all_matches(temp_grid): return True
                    temp_grid[r, c], temp_grid[r, c+1] = temp_grid[r, c+1], temp_grid[r, c]
                if r < self.GRID_HEIGHT - 1:
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
                    if self._find_all_matches(temp_grid): return True
                    temp_grid[r, c], temp_grid[r+1, c] = temp_grid[r+1, c], temp_grid[r, c]
        return False

    def _check_termination(self):
        return self.moves_left <= 0 or self.score >= self.TARGET_SCORE

    # --- Rendering Helpers ---
    
    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, self.grid_rect)
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                gem_type = self.grid[r, c]
                if gem_type != -1: self._draw_gem(gem_type, r, c)
                if (r, c) in self.last_matched_gems: self._draw_effect(r, c, self.COLOR_MATCH_FLASH)

        cx, cy = self.cursor_pos
        cursor_rect = pygame.Rect(self.grid_rect.left + cx * self.cell_size, self.grid_rect.top + cy * self.cell_size, self.cell_size, self.cell_size)
        s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA); s.fill(self.COLOR_CURSOR); self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, (255,255,255), cursor_rect, 2)

        if self.selected_gem_pos is not None:
            sx, sy = self.selected_gem_pos
            self._draw_effect(sy, sx, self.COLOR_SELECT, border_width=4)

    def _draw_gem(self, gem_type, r, c):
        center_x = int(self.grid_rect.left + (c + 0.5) * self.cell_size)
        center_y = int(self.grid_rect.top + (r + 0.5) * self.cell_size)
        radius = int(self.cell_size * 0.38)
        color = self.GEM_COLORS[gem_type % len(self.GEM_COLORS)]
        
        shape_type = gem_type % 3
        if shape_type == 0: # Circle
            pygame.gfxdraw.aacircle(self.screen, center_x, center_y, radius, color)
            pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, radius, color)
        elif shape_type == 1: # Square
            rect = pygame.Rect(center_x - radius, center_y - radius, radius * 2, radius * 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
        elif shape_type == 2: # Triangle
            points = [(center_x, center_y - radius), (center_x - radius, center_y + radius * 0.7), (center_x + radius, center_y + radius * 0.7)]
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        highlight_color = (255, 255, 255, 90)
        pygame.gfxdraw.filled_circle(self.screen, center_x - radius//2, center_y - radius//2, radius//3, highlight_color)

    def _draw_effect(self, r, c, color, border_width=0):
        rect = pygame.Rect(self.grid_rect.left + c * self.cell_size, self.grid_rect.top + r * self.cell_size, self.cell_size, self.cell_size)
        if border_width > 0:
            pygame.draw.rect(self.screen, color, rect, border_width, border_radius=6)
        else:
            s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA); s.fill(color); self.screen.blit(s, rect.topleft)

    def _render_ui(self):
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        moves_text = self.font_main.render(f"MOVES: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - moves_text.get_width() - 20, 20))

        goal_text = self.font_desc.render(f"GOAL: {self.TARGET_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(goal_text, (20, 50))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 180)); self.screen.blit(overlay, (0, 0))
            end_text_str, end_color = ("LEVEL CLEAR!", (100, 255, 100)) if self.score >= self.TARGET_SCORE else ("GAME OVER", (255, 100, 100))
            end_text = self.font_main.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    env.validate_implementation()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Gem Matcher")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: obs, info = env.reset()
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(2000)
            obs, info = env.reset()

        clock.tick(10)

    env.close()