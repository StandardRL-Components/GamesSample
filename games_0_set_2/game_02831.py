
# Generated: 2025-08-28T06:06:00.706647
# Source Brief: brief_02831.md
# Brief Index: 2831

        
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


class Particle:
    """A simple particle for visual effects."""
    def __init__(self, x, y, color, np_random):
        self.x = x
        self.y = y
        self.color = color
        self.np_random = np_random
        
        angle = self.np_random.uniform(0, 2 * math.pi)
        speed = self.np_random.uniform(2, 6)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        
        self.lifespan = self.np_random.uniform(20, 40)
        self.size = self.np_random.uniform(3, 8)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.lifespan -= 1
        self.size = max(0, self.size - 0.1)

    def draw(self, surface):
        if self.lifespan > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use ←→↑↓ to push rows/columns. "
        "Press SPACE to select the next column, SHIFT to select the next row."
    )

    game_description = (
        "A pixel puzzle game. Push rows and columns of pixels to match the target image "
        "before you run out of moves."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_DIM = 8
    MAX_MOVES = 50
    MAX_STEPS = 1000

    # --- Colors ---
    COLOR_BG = (20, 30, 40)
    COLOR_GRID_BG = (40, 50, 60)
    COLOR_UI_TEXT = (220, 220, 230)
    COLOR_SUCCESS = (100, 255, 100)
    COLOR_FAILURE = (255, 100, 100)
    COLOR_SELECTOR_BASE = (255, 255, 0)
    
    PALETTE = [
        (0, 0, 0, 0),  # Transparent for 0
        (230, 60, 60),    # Red
        (60, 230, 60),    # Green
        (60, 60, 230),    # Blue
        (230, 230, 60),   # Yellow
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
        
        # --- Fonts (no file loading) ---
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.remaining_moves = 0
        self.target_image_data = None
        self.current_grid = None
        self.selected_row = 0
        self.selected_col = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        self.last_match_count = 0
        self.completed_rows = None
        self.completed_cols = None
        self.particles = []

        # --- Grid Rendering Properties ---
        self.pixel_size = 40
        self.grid_size_px = self.GRID_DIM * self.pixel_size
        self.grid_top_left = (
            (self.SCREEN_WIDTH - self.grid_size_px) // 2,
            (self.SCREEN_HEIGHT - self.grid_size_px) // 2 + 20
        )
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.remaining_moves = self.MAX_MOVES
        
        self.target_image_data = self._create_target_image()
        self.current_grid = self._shuffle_grid(self.target_image_data.copy())
        
        self.selected_row = 0
        self.selected_col = 0
        self.prev_space_held = False
        self.prev_shift_held = False
        
        self.last_match_count = self._count_matching_pixels()
        self.completed_rows = set()
        self.completed_cols = set()
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.last_match_count = self._count_matching_pixels()
        
        # --- Handle Selection ---
        if space_held and not self.prev_space_held:
            self.selected_col = (self.selected_col + 1) % self.GRID_DIM
            # sfx: select_blip.wav
        if shift_held and not self.prev_shift_held:
            self.selected_row = (self.selected_row + 1) % self.GRID_DIM
            # sfx: select_blip.wav
            
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Handle Grid Push ---
        move_made = False
        if movement != 0 and self.remaining_moves > 0:
            move_made = True
            self.remaining_moves -= 1
            # sfx: push_swoosh.wav
            if movement == 1:  # Up
                self.current_grid[:, self.selected_col] = np.roll(self.current_grid[:, self.selected_col], -1)
            elif movement == 2:  # Down
                self.current_grid[:, self.selected_col] = np.roll(self.current_grid[:, self.selected_col], 1)
            elif movement == 3:  # Left
                self.current_grid[self.selected_row, :] = np.roll(self.current_grid[self.selected_row, :], -1)
            elif movement == 4:  # Right
                self.current_grid[self.selected_row, :] = np.roll(self.current_grid[self.selected_row, :], 1)

        self.steps += 1
        
        reward = self._calculate_reward(move_made)
        terminated = self._check_termination()

        if terminated:
            self.game_over = True
            if self.win:
                reward += 100
                self._spawn_win_particles()
                # sfx: win_jingle.wav
            else:
                reward -= 100
                # sfx: lose_buzzer.wav
        
        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _calculate_reward(self, move_made):
        if not move_made:
            return -0.01 # Small penalty for no-op to encourage action

        reward = 0
        
        # Continuous feedback for pixel matching
        current_match_count = self._count_matching_pixels()
        match_delta = current_match_count - self.last_match_count
        reward += match_delta * 0.1

        # Event-based reward for completing rows/columns
        for r in range(self.GRID_DIM):
            if r not in self.completed_rows and np.array_equal(self.current_grid[r, :], self.target_image_data[r, :]):
                reward += 5
                self.completed_rows.add(r)
                # sfx: row_complete.wav
        
        for c in range(self.GRID_DIM):
            if c not in self.completed_cols and np.array_equal(self.current_grid[:, c], self.target_image_data[:, c]):
                reward += 5
                self.completed_cols.add(c)
                # sfx: col_complete.wav

        return reward

    def _check_termination(self):
        # Win condition
        if self._count_matching_pixels() == self.GRID_DIM * self.GRID_DIM:
            self.win = True
            return True
        
        # Loss conditions
        if self.remaining_moves <= 0:
            return True
        if self.steps >= self.MAX_STEPS:
            return True
            
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # --- Draw main grid background ---
        grid_rect = pygame.Rect(self.grid_top_left, (self.grid_size_px, self.grid_size_px))
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, grid_rect)

        # --- Draw pixels ---
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                color_index = self.current_grid[r, c]
                if color_index > 0:
                    color = self.PALETTE[color_index]
                    px, py = self.grid_top_left
                    pixel_rect = pygame.Rect(
                        px + c * self.pixel_size + 1,
                        py + r * self.pixel_size + 1,
                        self.pixel_size - 2,
                        self.pixel_size - 2
                    )
                    pygame.draw.rect(self.screen, color, pixel_rect, border_radius=4)
        
        # --- Draw selector highlight ---
        pulse = (math.sin(self.steps * 0.2) + 1) / 2  # Varies between 0 and 1
        alpha = int(100 + pulse * 155)
        
        # Row selector
        row_y = self.grid_top_left[1] + self.selected_row * self.pixel_size
        row_surface = pygame.Surface((self.grid_size_px, self.pixel_size), pygame.SRCALPHA)
        row_surface.fill((*self.COLOR_SELECTOR_BASE, alpha))
        self.screen.blit(row_surface, (self.grid_top_left[0], row_y))

        # Column selector
        col_x = self.grid_top_left[0] + self.selected_col * self.pixel_size
        col_surface = pygame.Surface((self.pixel_size, self.grid_size_px), pygame.SRCALPHA)
        col_surface.fill((*self.COLOR_SELECTOR_BASE, alpha))
        self.screen.blit(col_surface, (col_x, self.grid_top_left[1]))

        # --- Update and draw particles ---
        self.particles = [p for p in self.particles if p.lifespan > 0]
        for p in self.particles:
            p.update()
            p.draw(self.screen)

    def _render_ui(self):
        # --- Target Image Preview ---
        self._render_text("Target", self.font_medium, (40, 30))
        preview_pixel_size = 10
        preview_top_left = (40, 60)
        for r in range(self.GRID_DIM):
            for c in range(self.GRID_DIM):
                color_index = self.target_image_data[r, c]
                if color_index > 0:
                    color = self.PALETTE[color_index]
                    pixel_rect = pygame.Rect(
                        preview_top_left[0] + c * preview_pixel_size,
                        preview_top_left[1] + r * preview_pixel_size,
                        preview_pixel_size,
                        preview_pixel_size
                    )
                    pygame.draw.rect(self.screen, color, pixel_rect)

        # --- Stats Display ---
        stats_x = 520
        self._render_text("MOVES", self.font_medium, (stats_x, 30))
        self._render_text(str(self.remaining_moves), self.font_large, (stats_x, 60))

        self._render_text("SCORE", self.font_medium, (stats_x, 120))
        self._render_text(str(int(self.score)), self.font_large, (stats_x, 150))
        
        match_percent = self._count_matching_pixels() / (self.GRID_DIM**2) * 100
        self._render_text("MATCH", self.font_medium, (stats_x, 210))
        self._render_text(f"{match_percent:.1f}%", self.font_large, (stats_x, 240))

        # --- Game Over Overlay ---
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.win:
                msg = "COMPLETE!"
                color = self.COLOR_SUCCESS
            else:
                msg = "GAME OVER"
                color = self.COLOR_FAILURE
            
            self._render_text(msg, self.font_large, (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20), color, center=True)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "remaining_moves": self.remaining_moves,
            "match_percentage": self._count_matching_pixels() / (self.GRID_DIM**2) * 100
        }

    # --- Helper Methods ---
    def _render_text(self, text, font, pos, color=COLOR_UI_TEXT, center=False):
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surface, text_rect)

    def _create_target_image(self):
        target = np.zeros((self.GRID_DIM, self.GRID_DIM), dtype=int)
        # Create a 'P' shape for 'Pixel'
        target[1:7, 2] = 1
        target[1, 3:5] = 1
        target[2, 5] = 1
        target[3, 5] = 1
        target[4, 3:5] = 1
        return target

    def _shuffle_grid(self, grid):
        # Apply a number of random pushes to create a solvable puzzle
        num_shuffles = self.np_random.integers(15, 25)
        for _ in range(num_shuffles):
            axis = self.np_random.integers(0, 2)  # 0 for row, 1 for col
            index = self.np_random.integers(0, self.GRID_DIM)
            direction = self.np_random.choice([-1, 1])
            if axis == 0: # Row push
                grid[index, :] = np.roll(grid[index, :], direction)
            else: # Col push
                grid[:, index] = np.roll(grid[:, index], direction)
        return grid

    def _count_matching_pixels(self):
        return np.sum(self.current_grid == self.target_image_data)
    
    def _spawn_win_particles(self):
        center_x = self.grid_top_left[0] + self.grid_size_px / 2
        center_y = self.grid_top_left[1] + self.grid_size_px / 2
        for _ in range(100):
            color = random.choice(self.PALETTE[1:]) # Don't pick transparent
            self.particles.append(Particle(center_x, center_y, color, self.np_random))

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Pixel Pusher")
    clock = pygame.time.Clock()

    action = [0, 0, 0] # no-op, released, released
    terminated = False

    print(GameEnv.user_guide)

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            # --- Get Keys for Action ---
            keys = pygame.key.get_pressed()
            movement = 0
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            space = 1 if keys[pygame.K_SPACE] else 0
            shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space, shift]

            # --- Step the Environment ---
            obs, reward, terminated, truncated, info = env.step(action)
            
            if reward != 0:
                print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}")
        
        # --- Rendering ---
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human playability

    pygame.quit()
    env.close()