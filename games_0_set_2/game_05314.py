
# Generated: 2025-08-28T04:38:30.361701
# Source Brief: brief_05314.md
# Brief Index: 5314

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a block and match it with adjacent same-colored blocks."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match 3 or more colored blocks to score points. Reach 1000 points to win, but watch out, you only have 100 moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 12
    GRID_HEIGHT = 8
    BLOCK_SIZE = 40
    GRID_OFFSET_X = (SCREEN_WIDTH - GRID_WIDTH * BLOCK_SIZE) // 2
    GRID_OFFSET_Y = (SCREEN_HEIGHT - GRID_HEIGHT * BLOCK_SIZE) + 20

    COLOR_BG = (20, 25, 40)
    COLOR_GRID_LINES = (40, 50, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    
    BLOCK_COLORS = [
        (0, 0, 0),       # 0: Empty (not used for drawing)
        (255, 80, 80),   # 1: Red
        (80, 255, 80),   # 2: Green
        (80, 150, 255),  # 3: Blue
        (255, 255, 80),  # 4: Yellow
        (200, 80, 255),  # 5: Purple
    ]
    NUM_COLORS = len(BLOCK_COLORS) - 1
    MIN_MATCH_SIZE = 3
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.moves_left = None
        self.game_over = None
        self.steps = None
        self.last_space_press = None
        self.particles = None
        self.rng = None
        self.win_state = None
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.grid = self.rng.integers(1, self.NUM_COLORS + 1, size=(self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8)
        
        while self._check_and_clear_initial_matches():
            pass

        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.score = 0
        self.moves_left = 100
        self.game_over = False
        self.win_state = None
        self.steps = 0
        self.last_space_press = False
        self.particles = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        
        movement = action[0]
        space_held = action[1] == 1
        
        # --- Handle Cursor Movement ---
        if movement == 1: self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 2: self.cursor_pos[1] = min(self.GRID_HEIGHT - 1, self.cursor_pos[1] + 1)
        elif movement == 3: self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 4: self.cursor_pos[0] = min(self.GRID_WIDTH - 1, self.cursor_pos[0] + 1)

        # --- Handle Block Selection (on key press, not hold) ---
        is_space_press = space_held and not self.last_space_press
        self.last_space_press = space_held

        if is_space_press:
            self.moves_left -= 1
            
            cursor_x, cursor_y = self.cursor_pos
            block_color_idx = self.grid[cursor_y, cursor_x]

            if block_color_idx != 0:
                connected_blocks = self._find_connected_blocks(cursor_x, cursor_y)
                
                if len(connected_blocks) >= self.MIN_MATCH_SIZE:
                    # Sound: block_match.wav
                    num_cleared = len(connected_blocks)
                    reward += num_cleared
                    self.score += num_cleared * 10

                    if num_cleared >= 4: reward += 10; self.score += 50
                    if num_cleared >= 5: reward += 20; self.score += 100
                    
                    block_rgb = self.BLOCK_COLORS[block_color_idx]
                    self._create_particles(connected_blocks, block_rgb)
                    for x, y in connected_blocks:
                        self.grid[y, x] = 0

                    self._apply_gravity_and_refill()
                else: # Sound: move_fail.wav
                    pass

        # --- Check Termination ---
        terminated = False
        if self.score >= 1000:
            self.score = 1000
            reward += 100
            terminated = True
            self.game_over = True
            self.win_state = "WIN" # Sound: win_jingle.wav
        elif self.moves_left <= 0:
            self.moves_left = 0
            reward -= 10
            terminated = True
            self.game_over = True
            self.win_state = "LOSE" # Sound: lose_sound.wav
        
        return (self._get_observation(), reward, terminated, False, self._get_info())
    
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
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
        }
        
    def _render_game(self):
        self._update_and_draw_particles()
        
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_idx = self.grid[y, x]
                if color_idx != 0:
                    self._draw_block(x, y, color_idx)
        
        self._draw_cursor()

    def _draw_block(self, grid_x, grid_y, color_idx):
        inset = 3
        screen_x = self.GRID_OFFSET_X + grid_x * self.BLOCK_SIZE
        screen_y = self.GRID_OFFSET_Y + grid_y * self.BLOCK_SIZE
        
        main_color = self.BLOCK_COLORS[color_idx]
        light_color = tuple(min(255, c + 50) for c in main_color)
        dark_color = tuple(max(0, c - 50) for c in main_color)
        
        rect = pygame.Rect(screen_x + inset, screen_y + inset, self.BLOCK_SIZE - 2*inset, self.BLOCK_SIZE - 2*inset)
        pygame.draw.rect(self.screen, main_color, rect, border_radius=4)
        
        pygame.draw.line(self.screen, light_color, (rect.left, rect.bottom - 1), (rect.left, rect.top), 2)
        pygame.draw.line(self.screen, light_color, (rect.left, rect.top), (rect.right - 1, rect.top), 2)
        pygame.draw.line(self.screen, dark_color, (rect.right - 1, rect.top + 1), (rect.right - 1, rect.bottom - 1), 2)
        pygame.draw.line(self.screen, dark_color, (rect.right - 1, rect.bottom - 1), (rect.left + 1, rect.bottom - 1), 2)

    def _draw_cursor(self):
        cursor_x, cursor_y = self.cursor_pos
        screen_x = self.GRID_OFFSET_X + cursor_x * self.BLOCK_SIZE
        screen_y = self.GRID_OFFSET_Y + cursor_y * self.BLOCK_SIZE
        
        pulse = (math.sin(self.steps * 0.25) + 1) / 2
        thickness = int(2 + pulse * 2)
        
        rect = pygame.Rect(screen_x, screen_y, self.BLOCK_SIZE, self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, thickness, border_radius=6)

    def _render_ui(self):
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
        self.screen.blit(score_text, score_rect)
        
        moves_text = self.font_medium.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        moves_rect = moves_text.get_rect(center=(self.SCREEN_WIDTH // 2, 60))
        self.screen.blit(moves_text, moves_rect)
        
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win_state == "WIN" else "GAME OVER"
            color = (100, 255, 100) if self.win_state == "WIN" else (255, 100, 100)
            
            end_text = self.font_large.render(message, True, color)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, end_rect)

    def _find_connected_blocks(self, start_x, start_y):
        target_color = self.grid[start_y, start_x]
        if target_color == 0: return []
        q = [(start_x, start_y)]; visited = set(q); connected = []
        while q:
            x, y = q.pop(0)
            connected.append((x, y))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and \
                   (nx, ny) not in visited and self.grid[ny, nx] == target_color:
                    visited.add((nx, ny)); q.append((nx, ny))
        return connected

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y, x] != 0:
                    self.grid[empty_row, x], self.grid[y, x] = self.grid[y, x], self.grid[empty_row, x]
                    empty_row -= 1
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] == 0:
                    self.grid[y, x] = self.rng.integers(1, self.NUM_COLORS + 1)

    def _check_and_clear_initial_matches(self):
        matches_found = False
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0:
                    connected = self._find_connected_blocks(x, y)
                    if len(connected) >= self.MIN_MATCH_SIZE:
                        matches_found = True
                        for pos_x, pos_y in connected:
                            self.grid[pos_y, pos_x] = self.rng.integers(1, self.NUM_COLORS + 1)
        return matches_found

    def _create_particles(self, positions, color):
        for x, y in positions:
            center_x = self.GRID_OFFSET_X + x * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
            center_y = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE + self.BLOCK_SIZE / 2
            for _ in range(10):
                angle = random.uniform(0, 2 * math.pi); speed = random.uniform(1, 4)
                vel = [math.cos(angle) * speed, math.sin(angle) * speed]
                lifespan = random.randint(20, 40); size = random.uniform(2, 5)
                self.particles.append({"pos": [center_x, center_y], "vel": vel, "lifespan": lifespan, "max_lifespan": lifespan, "color": color, "size": size})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p["pos"][0] += p["vel"][0]; p["pos"][1] += p["vel"][1]; p["lifespan"] -= 1
            if p["lifespan"] > 0:
                active_particles.append(p)
                alpha = int(255 * (p["lifespan"] / p["max_lifespan"]))
                size = int(p["size"])
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, (*p["color"], alpha), (size, size), size)
                self.screen.blit(temp_surf, (int(p["pos"][0]) - size, int(p["pos"][1]) - size))
        self.particles = active_particles

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
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")