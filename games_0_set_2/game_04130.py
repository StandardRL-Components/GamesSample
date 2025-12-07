
# Generated: 2025-08-28T01:30:34.527730
# Source Brief: brief_04130.md
# Brief Index: 4130

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move cursor. Space to select a fruit group. Shift to reshuffle the board."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Match cascading fruits in a grid-based frenzy to reach a high score before time runs out."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_AREA_HEIGHT = 360 # Leave space for UI at top
    
    CELL_SIZE = min((SCREEN_WIDTH - 40) // GRID_WIDTH, GRID_AREA_HEIGHT // GRID_HEIGHT)
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * CELL_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT * CELL_SIZE) // 2 + 20

    NUM_FRUIT_TYPES = 5
    MIN_MATCH_SIZE = 3
    WIN_SCORE = 5000
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS

    # Colors
    COLOR_BG = (25, 30, 35)
    COLOR_GRID = (50, 60, 70)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_CURSOR = (255, 255, 255)
    
    FRUIT_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 150, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 18)
        
        # Initialize state variables
        self.grid = None
        self.cursor_pos = None
        self.score = None
        self.steps = None
        self.time_left = None
        self.game_over = None
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []
        self.is_resolving_board = False
        self.combo_count = 0

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.time_left = self.MAX_STEPS
        self.game_over = False
        self.is_resolving_board = False
        self.combo_count = 0
        
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.last_space_held = False
        self.last_shift_held = False
        self.particles = []

        self._fill_board()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        reward = 0
        self.steps += 1
        self.time_left -= 1
        
        space_pressed = space_held and not self.last_space_held
        shift_pressed = shift_held and not self.last_shift_held

        self._update_particles()

        if not self.is_resolving_board:
            self._handle_player_input(movement, space_pressed, shift_pressed)
            if space_pressed or shift_pressed:
                reward += self._resolve_board_state()
        else:
            reward += self._resolve_board_state()

        self.last_space_held = space_held
        self.last_shift_held = shift_held
        
        terminated = self.time_left <= 0 or self.score >= self.WIN_SCORE
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_player_input(self, movement, space_pressed, shift_pressed):
        # Move cursor
        if movement == 1 and self.cursor_pos[1] > 0: self.cursor_pos[1] -= 1
        elif movement == 2 and self.cursor_pos[1] < self.GRID_HEIGHT - 1: self.cursor_pos[1] += 1
        elif movement == 3 and self.cursor_pos[0] > 0: self.cursor_pos[0] -= 1
        elif movement == 4 and self.cursor_pos[0] < self.GRID_WIDTH - 1: self.cursor_pos[0] += 1

        if shift_pressed:
            # sfx: shuffle
            self._fill_board()
            self.score = max(0, self.score - 10) # Penalty
            self.is_resolving_board = True # Trigger a check for auto-matches

        if space_pressed:
            group = self._find_group(self.cursor_pos[0], self.cursor_pos[1])
            if len(group) >= self.MIN_MATCH_SIZE:
                # sfx: match_start
                self.combo_count = 1
                self._clear_fruits(group)
                self.is_resolving_board = True
            else:
                # sfx: invalid_selection
                pass

    def _resolve_board_state(self):
        reward = 0
        
        cleared_fruits, match_reward = self._process_matches()
        reward += match_reward

        if cleared_fruits > 0:
            self._apply_gravity()
            self._refill_top_rows()
            self.combo_count += 1
            # Keep resolving if a match was made
        else:
            # Board is stable
            self.is_resolving_board = False
            self.combo_count = 0
            # Anti-softlock: reshuffle if no moves are possible
            if not self._has_valid_moves():
                self._fill_board()
                # sfx: reshuffle_auto
        return reward

    def _process_matches(self):
        all_matches = self._find_all_matches()
        if not all_matches:
            return 0, 0

        total_cleared = 0
        fruits_to_clear = set()
        for group in all_matches:
            for fruit_pos in group:
                fruits_to_clear.add(fruit_pos)
        
        self._clear_fruits(list(fruits_to_clear))
        total_cleared = len(fruits_to_clear)

        # Calculate reward
        reward = total_cleared # +1 per fruit
        if total_cleared >= 5: reward += 10 # Size bonus
        if total_cleared >= 20: reward += 20 # Large clear bonus
        reward += (self.combo_count - 1) * 10 # Combo bonus
        
        self.score += reward
        return total_cleared, reward

    def _fill_board(self):
        self.grid = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))
        while not self._has_valid_moves():
            self.grid = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1, size=(self.GRID_WIDTH, self.GRID_HEIGHT))

    def _find_group(self, x, y):
        if self.grid[x, y] == 0:
            return []
        
        target_type = self.grid[x, y]
        q = deque([(x, y)])
        visited = set([(x, y)])
        group = []

        while q:
            cx, cy = q.popleft()
            group.append((cx, cy))
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and \
                   (nx, ny) not in visited and self.grid[nx, ny] == target_type:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        return group

    def _find_all_matches(self):
        matches = []
        checked = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in checked and self.grid[x, y] != 0:
                    group = self._find_group(x, y)
                    for pos in group:
                        checked.add(pos)
                    if len(group) >= self.MIN_MATCH_SIZE:
                        matches.append(group)
        return matches

    def _has_valid_moves(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[x, y] != 0:
                    group = self._find_group(x, y)
                    if len(group) >= self.MIN_MATCH_SIZE:
                        return True
        return False

    def _clear_fruits(self, group):
        # sfx: fruit_pop
        for x, y in group:
            fruit_type = self.grid[x, y]
            if fruit_type > 0:
                self._create_particles(x, y, fruit_type)
                self.grid[x, y] = 0

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[x, y] != 0:
                    if y != empty_row:
                        self.grid[x, empty_row] = self.grid[x, y]
                        self.grid[x, y] = 0
                    empty_row -= 1
    
    def _refill_top_rows(self):
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.grid[x, y] == 0:
                    self.grid[x, y] = self.np_random.integers(1, self.NUM_FRUIT_TYPES + 1)

    def _create_particles(self, grid_x, grid_y, fruit_type):
        px = self.GRID_X_OFFSET + grid_x * self.CELL_SIZE + self.CELL_SIZE / 2
        py = self.GRID_Y_OFFSET + grid_y * self.CELL_SIZE + self.CELL_SIZE / 2
        color = self.FRUIT_COLORS[fruit_type - 1]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            lifetime = random.randint(15, 30)
            self.particles.append([px, py, vx, vy, lifetime, color])

    def _update_particles(self):
        self.particles = [p for p in self.particles if p[4] > 0]
        for p in self.particles:
            p[0] += p[2]
            p[1] += p[3]
            p[4] -= 1 # lifetime
            p[2] *= 0.98 # friction
            p[3] *= 0.98

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect, border_radius=5)

        # Draw fruits
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                fruit_type = self.grid[x, y]
                if fruit_type > 0:
                    self._draw_fruit(x, y, fruit_type)
        
        # Highlight selected group
        if not self.is_resolving_board:
            group = self._find_group(self.cursor_pos[0], self.cursor_pos[1])
            if len(group) >= self.MIN_MATCH_SIZE:
                for x, y in group:
                    rect = pygame.Rect(
                        self.GRID_X_OFFSET + x * self.CELL_SIZE,
                        self.GRID_Y_OFFSET + y * self.CELL_SIZE,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                    pygame.draw.rect(self.screen, (255, 255, 255, 60), rect, border_radius=8)

        # Draw cursor
        cursor_rect = pygame.Rect(
            self.GRID_X_OFFSET + self.cursor_pos[0] * self.CELL_SIZE,
            self.GRID_Y_OFFSET + self.cursor_pos[1] * self.CELL_SIZE,
            self.CELL_SIZE, self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, cursor_rect, width=3, border_radius=8)
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p[4] / 30.0))))
            color = (*p[5], alpha)
            size = max(1, int(6 * (p[4] / 30.0)))
            pygame.draw.circle(self.screen, color, (int(p[0]), int(p[1])), size)

    def _draw_fruit(self, x, y, fruit_type):
        cx = self.GRID_X_OFFSET + x * self.CELL_SIZE + self.CELL_SIZE // 2
        cy = self.GRID_Y_OFFSET + y * self.CELL_SIZE + self.CELL_SIZE // 2
        radius = int(self.CELL_SIZE * 0.4)
        color = self.FRUIT_COLORS[fruit_type - 1]
        
        # Shadow
        shadow_color = (max(0, color[0]-100), max(0, color[1]-100), max(0, color[2]-100))
        pygame.gfxdraw.filled_circle(self.screen, cx+2, cy+2, radius, shadow_color)
        
        # Main fruit body
        pygame.gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
        pygame.gfxdraw.aacircle(self.screen, cx, cy, radius, color)
        
        # Highlight
        h_color = (min(255, color[0]+80), min(255, color[1]+80), min(255, color[2]+80))
        pygame.gfxdraw.filled_circle(self.screen, cx - radius//3, cy - radius//3, radius//3, h_color)
        pygame.gfxdraw.aacircle(self.screen, cx - radius//3, cy - radius//3, radius//3, h_color)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 10))

        # Timer
        time_sec = self.time_left // self.FPS
        timer_color = self.COLOR_UI_TEXT if time_sec > 10 else self.COLOR_TIMER_WARN
        timer_text = self.font_main.render(f"TIME: {time_sec}", True, timer_color)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(timer_text, timer_rect)

        # Combo
        if self.combo_count > 1:
            combo_text = self.font_main.render(f"COMBO x{self.combo_count}!", True, self.FRUIT_COLORS[self.combo_count % len(self.FRUIT_COLORS)])
            combo_rect = combo_text.get_rect(center=(self.SCREEN_WIDTH / 2, 25))
            self.screen.blit(combo_text, combo_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left_seconds": self.time_left // self.FPS,
            "is_resolving": self.is_resolving_board,
            "combo": self.combo_count
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

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    
    running = True
    terminated = False
    
    # Create a display window
    pygame.display.set_caption("Fruit Frenzy")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))

    action = np.array([0, 0, 0]) # No-op, no space, no shift

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Player Controls ---
        keys = pygame.key.get_pressed()
        
        # Movement
        action[0] = 0 # No movement
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # Space and Shift
        action[1] = 1 if keys[pygame.K_SPACE] else 0
        action[2] = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            if reward != 0:
                print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            # Simple game over screen
            font = pygame.font.SysFont("Arial", 50, bold=True)
            msg = "YOU WON!" if info['score'] >= GameEnv.WIN_SCORE else "TIME'S UP!"
            text = font.render(msg, True, (255, 255, 0))
            text_rect = text.get_rect(center=(GameEnv.SCREEN_WIDTH/2, GameEnv.SCREEN_HEIGHT/2))
            display_screen.blit(text, text_rect)
            pygame.display.flip()
            
            # Wait for a key press to reset
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN:
                        obs, info = env.reset()
                        terminated = False
                        waiting_for_reset = False

        env.clock.tick(GameEnv.FPS)

    env.close()