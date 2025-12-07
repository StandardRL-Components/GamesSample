
# Generated: 2025-08-28T02:33:10.377941
# Source Brief: brief_04482.md
# Brief Index: 4482

        
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
        "Controls: ←→ to move block. ↓ to soft drop. Space to hard drop. ↑ does nothing."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Clear rows of matching colored blocks before they stack to the top. Progress through 3 stages of increasing speed."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    TOTAL_BLOCKS_TO_WIN = 50
    BLOCKS_PER_STAGE = [15, 15, 20] # Sums to 50
    STAGE_TIME_LIMIT = 60 # seconds

    # Colors
    COLOR_BG = (25, 25, 35)
    COLOR_GRID_BG = (40, 40, 55)
    COLOR_GRID_LINES = (60, 60, 80)
    COLOR_UI_TEXT = (220, 220, 240)
    BLOCK_COLORS = [
        (255, 80, 80),   # Red
        (80, 255, 80),   # Green
        (80, 120, 255),  # Blue
        (255, 255, 80),  # Yellow
        (200, 80, 255),  # Purple
    ]
    
    # --- Initialization ---
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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)

        self.grid_rect = self._calculate_grid_rect()
        self.cell_size = self.grid_rect.width // self.GRID_WIDTH
        
        self.render_mode = render_mode
        self.np_random = None

        # State variables are initialized in reset()
        self.reset()
        
        self.validate_implementation()

    def _calculate_grid_rect(self):
        # Center the grid, leaving space for UI
        grid_h = self.SCREEN_HEIGHT * 0.9
        cell_size = grid_h / self.GRID_HEIGHT
        grid_w = cell_size * self.GRID_WIDTH
        
        grid_x = (self.SCREEN_WIDTH - grid_w) / 2
        grid_y = (self.SCREEN_HEIGHT - grid_h) / 2 + 10 # Shift down a bit
        return pygame.Rect(grid_x, grid_y, grid_w, grid_h)

    # --- Gymnasium Core Methods ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed=seed)
        else:
            self.np_random = np.random.default_rng()

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.stage = 1
        self.base_fall_speed = 1.0 # cells per second
        self.blocks_placed_in_stage = 0
        self.total_blocks_placed = 0
        self.stage_timer = self.STAGE_TIME_LIMIT

        self.current_block = None
        self.next_block_color_index = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
        self._spawn_new_block()

        self.prev_space_held = False
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # --- Time and Step Management ---
        self.steps += 1
        delta_time = self.clock.tick(30) / 1000.0
        self.stage_timer -= delta_time
        reward = -0.01 # Penalty for existing

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Horizontal movement
        if movement == 3: # Left
            self._move_block(-1, 0)
        elif movement == 4: # Right
            self._move_block(1, 0)

        # Vertical movement (drop speed)
        fall_multiplier = 4.0 if movement == 2 else 1.0 # Soft drop
        
        # Hard drop on space press (rising edge)
        if space_held and not self.prev_space_held:
            # Sound effect: Hard drop thud
            while not self._check_collision(self.current_block['x'], self.current_block['y'] + 1):
                self.current_block['y'] += 1
            # Lock immediately
            reward += self._lock_block()
        else:
            # Normal fall
            self.current_block['y'] += self.base_fall_speed * fall_multiplier * delta_time
            if self._check_collision(self.current_block['x'], self.current_block['y']):
                # Sound effect: Block landing tap
                reward += self._lock_block()
        
        self.prev_space_held = space_held

        # --- Particle Update ---
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0] * delta_time
            p['pos'][1] += p['vel'][1] * delta_time
            p['vel'][1] += 50 * delta_time # Gravity
            p['life'] -= delta_time
            if p['life'] <= 0:
                self.particles.remove(p)

        # --- Termination Check ---
        terminated = self.game_over
        if not terminated:
            if self.stage_timer <= 0:
                reward -= 10 # Time out penalty
                terminated = True
            elif self.total_blocks_placed >= self.TOTAL_BLOCKS_TO_WIN:
                reward += 100 # Win game bonus
                terminated = True
            elif self.steps >= 10000:
                terminated = True
        
        self.game_over = terminated

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Game Logic Helpers ---
    def _spawn_new_block(self):
        self.current_block = {
            'x': self.GRID_WIDTH // 2 - 1,
            'y': 0.0,
            'color_index': self.next_block_color_index
        }
        self.next_block_color_index = self.np_random.integers(1, len(self.BLOCK_COLORS) + 1)
        
        # Check for game over on spawn
        if self._check_collision(self.current_block['x'], self.current_block['y']):
            self.game_over = True
            # Place the block anyway so it's visible on the final frame
            grid_y, grid_x = int(self.current_block['y']), int(self.current_block['x'])
            if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                self.grid[grid_y][grid_x] = self.current_block['color_index']

    def _move_block(self, dx, dy):
        if self.current_block and not self._check_collision(self.current_block['x'] + dx, self.current_block['y'] + dy):
            self.current_block['x'] += dx
            return True
        return False

    def _check_collision(self, x, y):
        grid_x = int(x)
        grid_y = int(y)
        
        if not (0 <= grid_x < self.GRID_WIDTH):
            return True # Wall collision
        if not (grid_y < self.GRID_HEIGHT):
            return True # Floor collision
        if grid_y >= 0 and self.grid[grid_y][grid_x] != 0:
            return True # Block collision
            
        return False

    def _lock_block(self):
        if not self.current_block: return 0

        grid_y, grid_x = int(self.current_block['y']), int(self.current_block['x'])
        
        # Ensure block is within grid bounds before placing
        if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
            self.grid[grid_y][grid_x] = self.current_block['color_index']
        else: # Block locked out of bounds (likely at top)
             self.game_over = True
             return -10 # Lose penalty

        if grid_y == 0: # Reached the top
            self.game_over = True
            return -10

        self.blocks_placed_in_stage += 1
        self.total_blocks_placed += 1
        
        reward = self._check_and_clear_rows()
        
        # Penalty for non-contributing placement
        if reward == 0: # Only penalize if no rows were cleared
            is_contributing = False
            color = self.current_block['color_index']
            # Check left
            if grid_x > 0 and self.grid[grid_y][grid_x - 1] == color:
                is_contributing = True
            # Check right
            if not is_contributing and grid_x < self.GRID_WIDTH - 1 and self.grid[grid_y][grid_x + 1] == color:
                is_contributing = True
            if not is_contributing:
                reward -= 0.2

        # Check for stage progression
        if not self.game_over and self.blocks_placed_in_stage >= self.BLOCKS_PER_STAGE[self.stage - 1]:
            self.stage += 1
            if self.stage > 3: # This case is handled by total_blocks_placed win condition
                pass 
            else:
                reward += 10 # Stage clear bonus
                self.blocks_placed_in_stage = 0
                self.base_fall_speed += 0.05
                self.stage_timer = self.STAGE_TIME_LIMIT
                # Sound effect: Stage complete fanfare

        if not self.game_over:
            self._spawn_new_block()

        return reward

    def _check_and_clear_rows(self):
        rows_to_clear = []
        for r in range(self.GRID_HEIGHT):
            first_color = self.grid[r][0]
            if first_color == 0:
                continue
            is_full_row = all(self.grid[r][c] == first_color for c in range(self.GRID_WIDTH))
            if is_full_row:
                rows_to_clear.append(r)

        if not rows_to_clear:
            return 0

        # Sound effect: Row clear chime
        for r in rows_to_clear:
            self._create_clear_particles(r, self.grid[r][0])

        # Shift rows down
        cleared_count = len(rows_to_clear)
        new_grid = np.zeros_like(self.grid)
        new_row_idx = self.GRID_HEIGHT - 1
        for r in range(self.GRID_HEIGHT - 1, -1, -1):
            if r not in rows_to_clear:
                new_grid[new_row_idx] = self.grid[r]
                new_row_idx -= 1
        self.grid = new_grid

        # Calculate reward
        self.score += cleared_count
        reward = cleared_count * 1.0 # +1 per row
        if cleared_count > 1:
            reward += 2.0 # Multi-clear bonus
            self.score += 2
        
        return reward

    def _create_clear_particles(self, row_index, color_index):
        # Sound effect: Particle burst
        row_y = self.grid_rect.top + row_index * self.cell_size + self.cell_size / 2
        color = self.BLOCK_COLORS[color_index - 1]
        for _ in range(30): # Number of particles
            px = self.grid_rect.left + self.np_random.random() * self.grid_rect.width
            vel_x = (self.np_random.random() - 0.5) * 150
            vel_y = (self.np_random.random() - 0.7) * 200
            self.particles.append({
                'pos': [px, row_y],
                'vel': [vel_x, vel_y],
                'life': self.np_random.random() * 0.5 + 0.3, # 0.3 to 0.8 seconds
                'color': color
            })

    # --- Rendering Helpers ---
    def _render_game(self):
        # Draw grid background and lines
        pygame.draw.rect(self.screen, self.COLOR_GRID_BG, self.grid_rect)
        for i in range(self.GRID_WIDTH + 1):
            x = self.grid_rect.left + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (x, self.grid_rect.top), (x, self.grid_rect.bottom))
        for i in range(self.GRID_HEIGHT + 1):
            y = self.grid_rect.top + i * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.grid_rect.left, y), (self.grid_rect.right, y))

        # Draw landed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] != 0:
                    self._draw_block(c, r, self.grid[r][c])
        
        # Draw falling block
        if self.current_block:
            self._draw_block(self.current_block['x'], self.current_block['y'], self.current_block['color_index'])
        
        # Draw particles
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / 0.5))))
            color_with_alpha = p['color'] + (alpha,)
            size = max(1, int(p['life'] * 8))
            s = pygame.Surface((size, size), pygame.SRCALPHA)
            pygame.draw.circle(s, color_with_alpha, (size // 2, size // 2), size // 2)
            self.screen.blit(s, (int(p['pos'][0] - size//2), int(p['pos'][1] - size//2)))


    def _draw_block(self, x, y, color_index):
        if color_index == 0: return
        
        block_x = self.grid_rect.left + x * self.cell_size
        block_y = self.grid_rect.top + y * self.cell_size
        
        base_color = self.BLOCK_COLORS[color_index - 1]
        light_color = tuple(min(255, c + 40) for c in base_color)
        dark_color = tuple(max(0, c - 40) for c in base_color)

        # Draw 3D-ish block
        outer_rect = pygame.Rect(block_x, block_y, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, dark_color, outer_rect)
        
        inner_rect = pygame.Rect(block_x + 2, block_y + 2, self.cell_size - 4, self.cell_size - 4)
        pygame.draw.rect(self.screen, base_color, inner_rect)

        # Add a small highlight for a "shiny" effect
        pygame.draw.line(self.screen, light_color, (block_x + 2, block_y + 2), (block_x + self.cell_size - 3, block_y + 2))
        pygame.draw.line(self.screen, light_color, (block_x + 2, block_y + 2), (block_x + 2, block_y + self.cell_size - 3))

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Stage and Timer
        stage_text = self.font_large.render(f"Stage: {self.stage}/3", True, self.COLOR_UI_TEXT)
        stage_rect = stage_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 20))
        self.screen.blit(stage_text, stage_rect)

        timer_text = self.font_small.render(f"Time: {max(0, int(self.stage_timer))}", True, self.COLOR_UI_TEXT)
        timer_rect = timer_text.get_rect(topright=(self.SCREEN_WIDTH - 20, stage_rect.bottom + 5))
        self.screen.blit(timer_text, timer_rect)

        # Next block preview
        next_text = self.font_small.render("Next:", True, self.COLOR_UI_TEXT)
        next_rect = next_text.get_rect(centerx=self.grid_rect.centerx, top=self.grid_rect.top - 50)
        self.screen.blit(next_text, next_rect)

        preview_x = (self.grid_rect.centerx - self.cell_size / 2) / self.cell_size
        preview_y = (next_rect.bottom + 5) / self.cell_size
        self._draw_block(preview_x, preview_y, self.next_block_color_index)

        # Game Over / Win Text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            
            if self.total_blocks_placed >= self.TOTAL_BLOCKS_TO_WIN:
                end_text_str = "YOU WIN!"
            else:
                end_text_str = "GAME OVER"
                
            end_text = self.font_large.render(end_text_str, True, (255, 255, 100))
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            overlay.blit(end_text, end_rect)
            self.screen.blit(overlay, (0, 0))

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Color Blocks")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # none
        space = 0
        shift = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        elif keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1
            
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

    env.close()