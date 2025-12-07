import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a "
        "square and clear all adjacent matching colors."
    )

    game_description = (
        "Clear the grid by matching adjacent colored squares. Larger matches give "
        "more points. Clear the entire board before the timer runs out to win!"
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_TIME = 60.0
        self.MAX_STEPS = int(self.MAX_TIME * self.FPS)

        self.GRID_COLS, self.GRID_ROWS = 5, 5
        self.CELL_SIZE = 60
        self.GRID_WIDTH = self.GRID_COLS * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_ROWS * self.CELL_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT) // 2

        # --- Colors ---
        self.COLOR_BG = (15, 20, 30)
        self.COLOR_GRID_LINES = (40, 50, 70)
        self.COLOR_EMPTY = (25, 30, 45)
        self.COLORS = [
            (255, 87, 87),    # Red
            (87, 255, 87),    # Green
            (87, 150, 255),   # Blue
            (255, 255, 87),   # Yellow
            (200, 87, 255),   # Purple
        ]
        self.COLOR_EMPTY_INDEX = -1

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_msg = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Game State (initialized in reset) ---
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.timer = None
        self.game_over = None
        self.win = None
        self.steps = None
        self.particles = None
        self.space_was_held = None
        self.last_movement_time = 0
        self.movement_cooldown = 4 # frames

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = self.np_random.integers(0, len(self.COLORS), size=(self.GRID_ROWS, self.GRID_COLS))
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        self.score = 0
        self.timer = self.MAX_TIME
        self.game_over = False
        self.win = False
        self.steps = 0
        self.particles = []
        self.space_was_held = False
        self.last_movement_time = 0

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.clock.tick(self.FPS)
        reward = 0
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.timer -= 1.0 / self.FPS
        self._update_particles()
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle cursor movement with a slight cooldown for better human playability
        if movement != 0 and (self.steps - self.last_movement_time) > self.movement_cooldown:
            self._handle_movement(movement)
            self.last_movement_time = self.steps
        
        # Handle activation on key press (rising edge)
        space_activated = space_held and not self.space_was_held
        if space_activated:
            reward += self._handle_activation()
        self.space_was_held = space_held

        # Check termination conditions
        if np.all(self.grid == self.COLOR_EMPTY_INDEX):
            self.game_over = True
            self.win = True
            reward += 100  # Win bonus
        elif self.timer <= 0 or self.steps >= self.MAX_STEPS:
            self.game_over = True
            self.win = False
            reward += -100 # Loss penalty

        terminated = self.game_over
        truncated = False # This env does not truncate based on time limit, it terminates.
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_movement(self, movement):
        if movement == 1:  # Up
            self.cursor_pos[1] -= 1
        elif movement == 2:  # Down
            self.cursor_pos[1] += 1
        elif movement == 3:  # Left
            self.cursor_pos[0] -= 1
        elif movement == 4:  # Right
            self.cursor_pos[0] += 1
        
        # Wrap around
        self.cursor_pos[0] %= self.GRID_COLS
        self.cursor_pos[1] %= self.GRID_ROWS

    def _handle_activation(self):
        x, y = self.cursor_pos
        target_color_idx = self.grid[y, x]

        if target_color_idx == self.COLOR_EMPTY_INDEX:
            # sfx: click_fail.wav
            return 0

        # Find all connected matching squares using BFS
        q = deque([(x, y)])
        visited = set([(x, y)])
        match_group = []

        while q:
            cx, cy = q.popleft()
            match_group.append((cx, cy))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_COLS and 0 <= ny < self.GRID_ROWS and \
                   (nx, ny) not in visited and self.grid[ny, nx] == target_color_idx:
                    visited.add((nx, ny))
                    q.append((nx, ny))

        if len(match_group) < 2:
            # sfx: click_fail.wav
            return 0

        # sfx: match_success.wav
        reward = 0
        
        # Check for "clear all of one color" bonus
        color_count_before = np.count_nonzero(self.grid == target_color_idx)
        if color_count_before == len(match_group):
            reward += 10

        # Clear matched squares
        for mx, my in match_group:
            self._create_particles(mx, my, self.COLORS[target_color_idx])
            self.grid[my, mx] = self.COLOR_EMPTY_INDEX
            reward += 1 # Base reward per square
            self.score += 1

        self._apply_gravity_and_refill()
        return reward

    def _apply_gravity_and_refill(self):
        for x in range(self.GRID_COLS):
            empty_slots = 0
            for y in range(self.GRID_ROWS - 1, -1, -1):
                if self.grid[y, x] == self.COLOR_EMPTY_INDEX:
                    empty_slots += 1
                elif empty_slots > 0:
                    self.grid[y + empty_slots, x] = self.grid[y, x]
                    self.grid[y, x] = self.COLOR_EMPTY_INDEX
            
            # Refill top empty slots
            for y in range(empty_slots):
                self.grid[y, x] = self.np_random.integers(0, len(self.COLORS))

    def _create_particles(self, grid_x, grid_y, color):
        center_x = self.GRID_X + grid_x * self.CELL_SIZE + self.CELL_SIZE // 2
        center_y = self.GRID_Y + grid_y * self.CELL_SIZE + self.CELL_SIZE // 2
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': vel,
                'color': color,
                'lifespan': lifespan,
                'max_lifespan': lifespan
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['lifespan'] -= 1
        self.particles = [p for p in self.particles if p['lifespan'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_grid()
        self._render_cursor()
        self._render_particles()
        self._render_ui()
        if self.game_over:
            self._render_game_over_message()
            
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_grid(self):
        for y in range(self.GRID_ROWS):
            for x in range(self.GRID_COLS):
                rect = pygame.Rect(
                    self.GRID_X + x * self.CELL_SIZE,
                    self.GRID_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE, self.CELL_SIZE
                )
                
                color_idx = self.grid[y, x]
                color = self.COLOR_EMPTY if color_idx == self.COLOR_EMPTY_INDEX else self.COLORS[color_idx]
                
                # Draw block with a border effect
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                # Add a subtle inner highlight for 3D effect
                if color_idx != self.COLOR_EMPTY_INDEX:
                    highlight_color = tuple(min(255, c + 40) for c in color)
                    inner_rect = rect.inflate(-8, -8)
                    pygame.draw.rect(self.screen, highlight_color, inner_rect, border_radius=4)

        # Draw grid lines over the blocks
        for i in range(self.GRID_COLS + 1):
            x = self.GRID_X + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT), 2)
        for i in range(self.GRID_ROWS + 1):
            y = self.GRID_Y + i * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH, y), 2)

    def _render_cursor(self):
        x, y = self.cursor_pos
        rect = pygame.Rect(
            self.GRID_X + x * self.CELL_SIZE,
            self.GRID_Y + y * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        
        # Pulsing alpha for the glow
        alpha = 128 + 127 * math.sin(self.steps * 0.2)
        glow_color = (255, 255, 255, alpha)
        
        # Create a temporary surface for the glowing rect
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(s, glow_color, s.get_rect(), width=4, border_radius=7)
        self.screen.blit(s, rect.topleft)

    def _render_particles(self):
        for p in self.particles:
            alpha = 255 * (p['lifespan'] / p['max_lifespan'])
            color = p['color']
            
            # Use gfxdraw for anti-aliased circles
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['lifespan'] * 0.2)
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*color, int(alpha)))
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*color, int(alpha)))

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, (220, 220, 240))
        self.screen.blit(score_surf, (20, 10))

        # Timer Bar
        timer_ratio = max(0, self.timer / self.MAX_TIME)
        bar_width = 200
        bar_height = 20
        bar_x = self.WIDTH - bar_width - 20
        bar_y = 15

        # Color interpolation
        if timer_ratio > 0.5:
            color = pygame.Color(87, 255, 87).lerp(pygame.Color(255, 255, 87), 1 - (timer_ratio - 0.5) * 2)
        else:
            color = pygame.Color(255, 255, 87).lerp(pygame.Color(255, 87, 87), 1 - timer_ratio * 2)

        pygame.draw.rect(self.screen, (40, 50, 70), (bar_x, bar_y, bar_width, bar_height), border_radius=5)
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, int(bar_width * timer_ratio), bar_height), border_radius=5)

    def _render_game_over_message(self):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        message = "YOU WIN!" if self.win else "TIME'S UP!"
        color = (100, 255, 100) if self.win else (255, 100, 100)
        
        msg_surf = self.font_msg.render(message, True, color)
        msg_rect = msg_surf.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
        
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(msg_surf, msg_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "timer": self.timer,
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]

        # Test reset to initialize state before other checks
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)

        # Test observation space (now that state is initialized)
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8

        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")