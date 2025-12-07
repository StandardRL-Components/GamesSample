
# Generated: 2025-08-27T18:52:53.719544
# Source Brief: brief_01982.md
# Brief Index: 1982

        
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
    """
    A puzzle game where the player clears groups of same-colored blocks from a grid
    to score points within a time limit. The game is designed for visual appeal
    and a satisfying gameplay experience.
    """
    metadata = {"render_modes": ["rgb_array"]}

    # --- User-facing descriptions ---
    user_guide = (
        "Controls: Use arrow keys to move the selector. Press space to clear the "
        "selected group of connected, same-colored blocks."
    )
    game_description = (
        "Connect and clear groups of same-colored blocks in this fast-paced puzzle game. "
        "Score as many points as you can before the timer runs out!"
    )

    # --- Frame advance mode ---
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_COLS = 10
    GRID_ROWS = 10
    BLOCK_SIZE = 36
    GRID_WIDTH = GRID_COLS * BLOCK_SIZE
    GRID_HEIGHT = GRID_ROWS * BLOCK_SIZE
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT) // 2
    
    FPS = 30
    GAME_DURATION_SECONDS = 60
    MAX_STEPS = GAME_DURATION_SECONDS * FPS
    MIN_CLEAR_GROUP = 2 # Minimum blocks required to form a clearable group

    # --- Colors ---
    COLOR_BG = (25, 35, 45)
    COLOR_GRID_LINES = (45, 55, 65)
    COLOR_CURSOR = (255, 255, 255)
    COLOR_TEXT = (230, 230, 230)
    COLOR_TEXT_SHADOW = (10, 10, 10)
    
    BLOCK_COLORS = [
        (0, 0, 0), # 0: Empty
        (227, 87, 73),   # 1: Red
        (87, 199, 106),  # 2: Green
        (73, 142, 227),  # 3: Blue
        (227, 211, 73),  # 4: Yellow
        (181, 73, 227),  # 5: Purple
    ]

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
        
        self.font_large = pygame.font.SysFont("sans-serif", 48, bold=True)
        self.font_medium = pygame.font.SysFont("sans-serif", 24, bold=True)
        
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.cursor_pos = [0, 0]
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.last_move_step = 0
        
        self.np_random = None # Will be initialized in reset
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             self.np_random = np.random.default_rng(seed=seed)
        else:
             self.np_random = np.random.default_rng()

        self.grid = self.np_random.integers(1, len(self.BLOCK_COLORS), size=(self.GRID_ROWS, self.GRID_COLS))
        self.cursor_pos = [self.GRID_COLS // 2, self.GRID_ROWS // 2]
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.prev_space_held = False
        self.last_move_step = 0
        self.particles.clear()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        if not self.game_over:
            self.steps += 1
            
            # --- Handle Input ---
            self._handle_movement(movement)
            if space_held and not self.prev_space_held:
                reward = self._clear_blocks()
            
            self._update_particles()
        
        self.prev_space_held = space_held
        terminated = self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            # No special terminal reward, score is the objective
            
        return (
            self._get_observation(),
            float(reward),
            terminated,
            False,
            self._get_info()
        )

    def _handle_movement(self, movement):
        # Cooldown to prevent cursor from moving too fast
        if movement != 0 and self.steps > self.last_move_step + 5:
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            if dx != 0 or dy != 0:
                self.cursor_pos[0] = (self.cursor_pos[0] + dx) % self.GRID_COLS
                self.cursor_pos[1] = (self.cursor_pos[1] + dy) % self.GRID_ROWS
                self.last_move_step = self.steps
                # sfx: cursor_move.wav

    def _clear_blocks(self):
        cx, cy = self.cursor_pos
        if self.grid[cy, cx] == 0:
            return 0

        connected_blocks = self._find_connected_blocks(cx, cy)
        
        if len(connected_blocks) >= self.MIN_CLEAR_GROUP:
            # sfx: block_clear.wav
            for r, c in connected_blocks:
                color = self.BLOCK_COLORS[self.grid[r, c]]
                self._spawn_particles(c, r, color)
                self.grid[r, c] = 0
            
            self._apply_gravity_and_refill()
            
            # Reward is proportional to the square of the number of blocks cleared
            # This encourages larger groups.
            reward = len(connected_blocks)
            self.score += reward
            return reward
        
        # sfx: invalid_move.wav
        return 0

    def _find_connected_blocks(self, start_c, start_r):
        target_color = self.grid[start_r, start_c]
        if target_color == 0:
            return []

        q = [(start_r, start_c)]
        visited = set(q)
        connected = []

        while q:
            r, c = q.pop(0)
            connected.append((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and \
                   (nr, nc) not in visited and self.grid[nr, nc] == target_color:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return connected

    def _apply_gravity_and_refill(self):
        for c in range(self.GRID_COLS):
            # Collect existing blocks in the column
            existing_blocks = [self.grid[r, c] for r in range(self.GRID_ROWS) if self.grid[r, c] != 0]
            
            # Create new blocks to fill the gaps
            num_new_blocks = self.GRID_ROWS - len(existing_blocks)
            new_blocks = self.np_random.integers(1, len(self.BLOCK_COLORS), size=num_new_blocks).tolist()
            
            # Combine and update the column
            new_column = new_blocks + existing_blocks
            for r in range(self.GRID_ROWS):
                self.grid[r, c] = new_column[r]

    def _spawn_particles(self, c, r, color):
        px = self.GRID_X_OFFSET + c * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        py = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE + self.BLOCK_SIZE // 2
        
        for _ in range(10): # Spawn 10 particles per block
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            radius = self.np_random.uniform(2, 5)
            self.particles.append({'pos': [px, py], 'vel': vel, 'color': color, 'radius': radius, 'lifespan': lifespan})
            
    def _update_particles(self):
        self.particles = [p for p in self.particles if p['lifespan'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['radius'] *= 0.95
            p['lifespan'] -= 1

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_ROWS + 1):
            y = self.GRID_Y_OFFSET + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH, y))
        for c in range(self.GRID_COLS + 1):
            x = self.GRID_X_OFFSET + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT))

        # Draw blocks
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                color_index = self.grid[r, c]
                if color_index != 0:
                    self._draw_block(c, r, self.BLOCK_COLORS[color_index])

        # Draw cursor
        cursor_x = self.GRID_X_OFFSET + self.cursor_pos[0] * self.BLOCK_SIZE
        cursor_y = self.GRID_Y_OFFSET + self.cursor_pos[1] * self.BLOCK_SIZE
        pygame.draw.rect(self.screen, self.COLOR_CURSOR, (cursor_x, cursor_y, self.BLOCK_SIZE, self.BLOCK_SIZE), 3)

        # Draw particles
        for p in self.particles:
            pos = (int(p['pos'][0]), int(p['pos'][1]))
            radius = int(p['radius'])
            if radius > 0:
                alpha = max(0, min(255, int(255 * (p['lifespan'] / 30.0))))
                # Use gfxdraw for anti-aliased, alpha-blended circles
                pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, (*p['color'], alpha))
                pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, (*p['color'], alpha))
    
    def _draw_block(self, c, r, color):
        rect = pygame.Rect(
            self.GRID_X_OFFSET + c * self.BLOCK_SIZE,
            self.GRID_Y_OFFSET + r * self.BLOCK_SIZE,
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        # Main color
        pygame.draw.rect(self.screen, color, rect)
        
        # 3D effect: highlight and shadow
        highlight_color = tuple(min(255, val + 40) for val in color)
        shadow_color = tuple(max(0, val - 40) for val in color)
        
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright, 2)
        pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft, 2)
        pygame.draw.line(self.screen, shadow_color, rect.bottomleft, rect.bottomright, 2)
        pygame.draw.line(self.screen, shadow_color, rect.topright, rect.bottomright, 2)
        
    def _render_text(self, text, font, x, y, color=COLOR_TEXT, shadow_color=COLOR_TEXT_SHADOW):
        shadow_surf = font.render(text, True, shadow_color)
        self.screen.blit(shadow_surf, (x + 2, y + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))

    def _render_ui(self):
        # Score
        self._render_text(f"SCORE: {self.score}", self.font_medium, 20, 10)
        
        # Timer
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        self._render_text(f"TIME: {time_left:.1f}", self.font_medium, self.SCREEN_WIDTH - 150, 10)

        # Game Over screen
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            self._render_text("TIME'S UP!", self.font_large, 
                              self.SCREEN_WIDTH // 2 - self.font_large.size("TIME'S UP!")[0] // 2, 
                              self.SCREEN_HEIGHT // 2 - 50)
            
            final_score_text = f"FINAL SCORE: {self.score}"
            self._render_text(final_score_text, self.font_medium, 
                              self.SCREEN_WIDTH // 2 - self.font_medium.size(final_score_text)[0] // 2, 
                              self.SCREEN_HEIGHT // 2 + 10)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "time_left": max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    done = False
    
    # --- Pygame setup for manual play ---
    pygame.display.set_caption("Block Clear")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    running = True

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward > 0:
            print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward}")

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # --- Display the observation from the environment ---
        # The observation is (H, W, C), but pygame blits (W, H) surfaces.
        # So we need to transpose it back.
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    env.close()