
# Generated: 2025-08-27T22:30:31.649563
# Source Brief: brief_03148.md
# Brief Index: 3148

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: Arrow keys to move the cursor. Spacebar to select a block and clear matching groups."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Clear the board by selecting groups of 3 or more same-colored blocks. You have a limited number of moves!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 8
        self.MAX_MOVES = 20
        self.MIN_MATCH = 3
        
        # --- Visuals ---
        self.BLOCK_SIZE = int(self.SCREEN_HEIGHT * 0.09)
        self.GRID_LINE_WIDTH = 1
        self.PARTICLE_LIFESPAN = 30
        self.PARTICLE_COUNT = 15

        self.GRID_AREA_WIDTH = self.GRID_WIDTH * self.BLOCK_SIZE
        self.GRID_AREA_HEIGHT = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.GRID_OFFSET_X = (self.SCREEN_WIDTH - self.GRID_AREA_WIDTH) // 2
        self.GRID_OFFSET_Y = (self.SCREEN_HEIGHT - self.GRID_AREA_HEIGHT) - 10 # Place slightly lower

        # --- Colors ---
        self.COLOR_BG = (25, 35, 45)
        self.COLOR_GRID = (45, 55, 65)
        self.COLOR_TEXT = (230, 230, 230)
        self.COLORS = {
            1: (227, 99, 135),   # Red
            2: (107, 214, 153),  # Green
            3: (99, 155, 227),   # Blue
            4: (245, 225, 122),  # Yellow
            5: (179, 140, 226),  # Purple
        }
        self.NUM_COLORS = len(self.COLORS)
        
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
        self.font_ui = pygame.font.Font(None, 36)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # --- State Variables ---
        self.grid = None
        self.cursor_pos = None
        self.moves_left = None
        self.score = None
        self.game_over = None
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        self.steps = 0
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.cursor_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self._generate_board()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = 0
        terminated = False

        self._update_particles()
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # Handle Shift (interpreted as a self-inflicted loss for RL)
        if shift_held and not self.prev_shift_held:
            self.game_over = True
            terminated = True
            reward = -50.0  # Loss penalty
            self.prev_shift_held = shift_held
            return self._get_observation(), reward, terminated, False, self._get_info()
        self.prev_shift_held = shift_held

        # Handle cursor movement
        if movement != 0:
            dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][movement]
            self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_WIDTH
            self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_HEIGHT

        # Handle block selection on space press
        if space_held and not self.prev_space_held:
            # SFX: click
            reward += self._process_selection()
        self.prev_space_held = space_held
        
        # Check termination conditions
        if self.moves_left <= 0 and not self._is_board_clear():
            self.game_over = True
            terminated = True
            reward += -50.0  # Loss penalty

        if self._is_board_clear():
            self.game_over = True
            terminated = True
            reward += 100.0  # Win bonus
            self.score += 1000 # Bonus score for winning

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _process_selection(self):
        self.moves_left -= 1
        x, y = self.cursor_pos
        
        connected = self._find_connected_blocks(x, y)
        
        if len(connected) >= self.MIN_MATCH:
            # SFX: match_success
            num_cleared = len(connected)
            reward = float(num_cleared)  # +1 per block
            if num_cleared > 4:
                reward += 5.0 # Bonus for large clear
            
            self.score += num_cleared * num_cleared # Exponential score for bigger matches
            
            block_color_id = self.grid[y][x]
            block_color_rgb = self.COLORS[block_color_id]
            
            for bx, by in connected:
                self.grid[by][bx] = 0  # 0 for empty
                self._create_particles(bx, by, block_color_rgb)
            
            self._apply_gravity()
            self._fill_top_rows()
            return reward
        else:
            # SFX: match_fail
            return -0.1 # Penalty for invalid move

    def _find_connected_blocks(self, start_x, start_y):
        if self.grid[start_y][start_x] == 0:
            return set()

        target_color = self.grid[start_y][start_x]
        q = deque([(start_x, start_y)])
        visited = set([(start_x, start_y)])
        
        while q:
            x, y = q.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.GRID_WIDTH and 0 <= ny < self.GRID_HEIGHT and (nx, ny) not in visited:
                    if self.grid[ny][nx] == target_color:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return visited

    def _apply_gravity(self):
        for x in range(self.GRID_WIDTH):
            empty_row = self.GRID_HEIGHT - 1
            for y in range(self.GRID_HEIGHT - 1, -1, -1):
                if self.grid[y][x] != 0:
                    self.grid[empty_row][x], self.grid[y][x] = self.grid[y][x], self.grid[empty_row][x]
                    empty_row -= 1

    def _fill_top_rows(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y][x] == 0:
                    self.grid[y][x] = self.np_random.integers(1, self.NUM_COLORS + 1)

    def _is_board_clear(self):
        return all(self.grid[y][x] == 0 for y in range(self.GRID_HEIGHT) for x in range(self.GRID_WIDTH))

    def _generate_board(self):
        while True:
            self.grid = [[self.np_random.integers(1, self.NUM_COLORS + 1) for _ in range(self.GRID_WIDTH)] for _ in range(self.GRID_HEIGHT)]
            if self._has_valid_moves():
                break

    def _has_valid_moves(self):
        visited_for_check = set()
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if (x, y) not in visited_for_check:
                    connected = self._find_connected_blocks(x, y)
                    if len(connected) >= self.MIN_MATCH:
                        return True
                    visited_for_check.update(connected)
        return False

    def _create_particles(self, grid_x, grid_y, color):
        cx = self.GRID_OFFSET_X + grid_x * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        cy = self.GRID_OFFSET_Y + grid_y * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        for _ in range(self.PARTICLE_COUNT):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({'pos': [cx, cy], 'vel': vel, 'life': self.PARTICLE_LIFESPAN, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_OFFSET_X + x * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, self.GRID_OFFSET_Y), (px, self.GRID_OFFSET_Y + self.GRID_AREA_HEIGHT), self.GRID_LINE_WIDTH)
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_OFFSET_X, py), (self.GRID_OFFSET_X + self.GRID_AREA_WIDTH, py), self.GRID_LINE_WIDTH)

        # Draw blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                color_id = self.grid[y][x]
                if color_id != 0:
                    px = self.GRID_OFFSET_X + x * self.BLOCK_SIZE
                    py = self.GRID_OFFSET_Y + y * self.BLOCK_SIZE
                    color = self.COLORS[color_id]
                    
                    # Block with 3D effect
                    border_color = tuple(max(0, c - 40) for c in color)
                    pygame.draw.rect(self.screen, border_color, (px, py, self.BLOCK_SIZE, self.BLOCK_SIZE))
                    inset = 3
                    pygame.draw.rect(self.screen, color, (px + inset, py + inset, self.BLOCK_SIZE - inset*2, self.BLOCK_SIZE - inset*2))

        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / self.PARTICLE_LIFESPAN))))
            color = p['color']
            radius = int(max(1, 5 * (p['life'] / self.PARTICLE_LIFESPAN)))
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, (*color, alpha))

        # Draw cursor
        cursor_x, cursor_y = self.cursor_pos
        cx = self.GRID_OFFSET_X + cursor_x * self.BLOCK_SIZE
        cy = self.GRID_OFFSET_Y + cursor_y * self.BLOCK_SIZE
        
        # Pulsating glow effect
        pulse = (math.sin(self.steps * 0.3) + 1) / 2 # 0 to 1
        alpha = 70 + 80 * pulse
        size_inflation = int(4 * pulse)
        
        cursor_rect = pygame.Rect(cx - size_inflation, cy - size_inflation, self.BLOCK_SIZE + size_inflation*2, self.BLOCK_SIZE + size_inflation*2)
        
        # Create a temporary surface for transparency
        s = pygame.Surface((cursor_rect.width, cursor_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(s, (255, 255, 255, alpha), s.get_rect(), border_radius=5)
        self.screen.blit(s, cursor_rect.topleft)
        pygame.draw.rect(self.screen, (255, 255, 255), cursor_rect, 2, border_radius=5)

    def _render_ui(self):
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 20))
        
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(score_text, score_rect)
        
    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        
        win_condition = self._is_board_clear()
        message = "YOU WIN!" if win_condition else "GAME OVER"
        color = (100, 255, 150) if win_condition else (255, 100, 100)
        
        text = self.font_game_over.render(message, True, color)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        
        overlay.blit(text, text_rect)
        self.screen.blit(overlay, (0, 0))

    def _get_info(self):
        return {
            "score": self.score,
            "moves_left": self.moves_left,
            "cursor_pos": list(self.cursor_pos),
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
    # To run and play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Block Breaker Gym Env")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Blit the observation from the environment to the display screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
        
        clock.tick(30) # Limit frame rate for human play

    env.close()