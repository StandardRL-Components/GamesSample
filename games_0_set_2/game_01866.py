
# Generated: 2025-08-27T18:32:17.317340
# Source Brief: brief_01866.md
# Brief Index: 1866

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↓ for soft drop, ↑ to rotate. "
        "Space for hard drop. Fill rows to score."
    )

    game_description = (
        "A fast-paced puzzle game where you place falling blocks to clear lines. "
        "Clear 5 lines to win, but you only have 200 moves!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.BLOCK_SIZE = 18
        self.MAX_STEPS = 1000
        self.MAX_MOVES = 200
        self.WIN_CONDITION_LINES = 5

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_GHOST = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_TEXT_ACCENT = (255, 215, 0)
        self.COLOR_CLEAR_ANIM = (255, 255, 255)
        self.BLOCK_COLORS = [
            (0, 240, 240),  # I - Cyan
            (0, 0, 240),    # J - Blue
            (240, 160, 0),  # L - Orange
            (240, 240, 0),  # O - Yellow
            (0, 240, 0),    # S - Green
            (160, 0, 240),  # T - Purple
            (240, 0, 0),    # Z - Red
        ]

        # --- Tetromino Shapes ---
        self.TETROMINOES = {
            0: [[1, 1, 1, 1]],  # I
            1: [[1, 0, 0], [1, 1, 1]],  # J
            2: [[0, 0, 1], [1, 1, 1]],  # L
            3: [[1, 1], [1, 1]],  # O
            4: [[0, 1, 1], [1, 1, 0]],  # S
            5: [[0, 1, 0], [1, 1, 1]],  # T
            6: [[1, 1, 0], [0, 1, 1]],  # Z
        }
        
        # --- Scoring ---
        self.LINE_CLEAR_REWARDS = {1: 1, 2: 3, 3: 5, 4: 8} # Points per lines cleared at once

        # --- Gymnasium Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        self.font_title = pygame.font.SysFont("Consolas", 20, bold=True)

        # --- Grid Position ---
        self.grid_render_x = (self.WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.grid_render_y = (self.HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2
        
        # --- State Variables ---
        self.grid = None
        self.current_block_idx = None
        self.current_block_shape = None
        self.current_block_pos = None
        self.next_block_idx = None
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.moves_left = 0
        self.terminated = False
        self.lines_to_clear_anim = []
        self._current_reward = 0
        self.random_generator = None

        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.random_generator = random.Random(seed)
        else:
            self.random_generator = random.Random()

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.moves_left = self.MAX_MOVES
        self.terminated = False
        self.lines_to_clear_anim = []

        self._spawn_block()
        self._spawn_block() # Call twice to populate current and next

        return self._get_observation(), self._get_info()

    def step(self, action):
        self.steps += 1
        self._current_reward = 0
        self.lines_to_clear_anim = [] # Clear previous step's animation

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        
        self.moves_left -= 1

        if not self.terminated:
            if space_held:
                self._hard_drop()
            else:
                self._handle_movement(movement)
        
        # Check termination conditions
        is_win = self.lines_cleared >= self.WIN_CONDITION_LINES
        is_loss_moves = self.moves_left <= 0
        is_loss_steps = self.steps >= self.MAX_STEPS

        if is_win and not self.terminated:
            self._current_reward += 100
            self.terminated = True
        elif (is_loss_moves or self.terminated) and not is_win: # self.terminated can be set by top-out
            self._current_reward -= 100
            self.terminated = True
        elif is_loss_steps and not self.terminated:
             self.terminated = True

        self.score += self._current_reward

        return (
            self._get_observation(),
            self._current_reward,
            self.terminated,
            False,
            self._get_info(),
        )

    def _spawn_block(self):
        self.current_block_idx = self.next_block_idx
        self.current_block_shape = self.TETROMINOES.get(self.current_block_idx)
        
        self.next_block_idx = self.random_generator.randint(0, len(self.TETROMINOES) - 1)
        
        if self.current_block_shape is not None:
            self.current_block_pos = [self.GRID_WIDTH // 2 - len(self.current_block_shape[0]) // 2, 0]
            if self._check_collision(self.current_block_shape, self.current_block_pos):
                self.terminated = True # Game over: top out

    def _handle_movement(self, movement):
        # 0=noop, 1=up/rotate, 2=down, 3=left, 4=right
        if movement == 1: # Rotate
            self._rotate_block()
        elif movement == 3: # Left
            self._move(-1, 0)
        elif movement == 4: # Right
            self._move(1, 0)
        
        # Soft drop is default action for noop or down
        if movement == 0 or movement == 2:
            self._move(0, 1, is_soft_drop=True)

    def _move(self, dx, dy, is_soft_drop=False):
        new_pos = [self.current_block_pos[0] + dx, self.current_block_pos[1] + dy]
        if not self._check_collision(self.current_block_shape, new_pos):
            self.current_block_pos = new_pos
        elif dy > 0: # Collision while moving down
            self._lock_block()
            if is_soft_drop:
                self._current_reward += 0.01 # Small reward for soft dropping

    def _hard_drop(self):
        # sound: block_slam.wav
        dy = 0
        while not self._check_collision(self.current_block_shape, [self.current_block_pos[0], self.current_block_pos[1] + dy + 1]):
            dy += 1
        self.current_block_pos[1] += dy
        self._current_reward += 0.02 # Small reward for hard dropping
        self._lock_block()

    def _rotate_block(self):
        # sound: block_rotate.wav
        rotated_shape = list(zip(*self.current_block_shape[::-1]))
        
        # Wall kick logic
        for dx in [0, 1, -1, 2, -2]: # Basic wall kick checks
            if not self._check_collision(rotated_shape, [self.current_block_pos[0] + dx, self.current_block_pos[1]]):
                self.current_block_shape = rotated_shape
                self.current_block_pos[0] += dx
                return

    def _lock_block(self):
        # sound: block_land.wav
        for r, row in enumerate(self.current_block_shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = self.current_block_pos[0] + c
                    grid_y = self.current_block_pos[1] + r
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_block_idx + 1
        
        self._current_reward += 0.1 # Reward for placing a block
        self._clear_lines()
        self._spawn_block()

    def _clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.grid) if np.all(row > 0)]
        
        if lines_to_clear:
            # sound: line_clear.wav
            num_cleared = len(lines_to_clear)
            self.lines_cleared += num_cleared
            self.lines_to_clear_anim = lines_to_clear # For rendering flash
            
            # Add reward
            reward = self.LINE_CLEAR_REWARDS.get(num_cleared, 0)
            self._current_reward += reward

            # Remove rows and add new empty rows at the top
            self.grid = np.delete(self.grid, lines_to_clear, axis=0)
            new_rows = np.zeros((num_cleared, self.GRID_WIDTH), dtype=int)
            self.grid = np.vstack((new_rows, self.grid))

    def _check_collision(self, shape, pos):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = pos[0] + c, pos[1] + r
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return True # Out of bounds
                    if self.grid[grid_y, grid_x] > 0:
                        return True # Collides with another block
        return False

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
            "lines_cleared": self.lines_cleared,
            "moves_left": self.moves_left,
        }

    def _render_game(self):
        # Draw grid background
        pygame.draw.rect(
            self.screen, self.COLOR_GRID,
            (self.grid_render_x, self.grid_render_y, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        )

        # Draw placed blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] > 0:
                    self._draw_block(c, r, self.grid[r, c] - 1)
        
        # Draw line clear animation
        if self.lines_to_clear_anim:
            for r in self.lines_to_clear_anim:
                pygame.draw.rect(
                    self.screen, self.COLOR_CLEAR_ANIM,
                    (self.grid_render_x, self.grid_render_y + r * self.BLOCK_SIZE, self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE),
                )

        # Draw current block if it exists
        if self.current_block_shape:
            # Draw ghost piece
            ghost_pos = list(self.current_block_pos)
            while not self._check_collision(self.current_block_shape, [ghost_pos[0], ghost_pos[1] + 1]):
                ghost_pos[1] += 1
            self._draw_block_shape(self.current_block_shape, ghost_pos, self.current_block_idx, is_ghost=True)

            # Draw the actual falling block
            self._draw_block_shape(self.current_block_shape, self.current_block_pos, self.current_block_idx)
        
        # Draw grid border
        pygame.draw.rect(
            self.screen, self.COLOR_TEXT,
            (self.grid_render_x, self.grid_render_y, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE), 2
        )

    def _draw_block(self, grid_c, grid_r, color_idx, is_ghost=False):
        x = self.grid_render_x + grid_c * self.BLOCK_SIZE
        y = self.grid_render_y + grid_r * self.BLOCK_SIZE
        color = self.BLOCK_COLORS[color_idx]

        if is_ghost:
            pygame.draw.rect(self.screen, self.COLOR_GHOST, (x, y, self.BLOCK_SIZE, self.BLOCK_SIZE), 1)
        else:
            main_rect = pygame.Rect(x, y, self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, color, main_rect)
            # Add a 3D effect
            darker_color = tuple(max(0, c - 50) for c in color)
            lighter_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.line(self.screen, lighter_color, (x, y), (x + self.BLOCK_SIZE - 1, y), 2)
            pygame.draw.line(self.screen, lighter_color, (x, y), (x, y + self.BLOCK_SIZE - 1), 2)
            pygame.draw.line(self.screen, darker_color, (x + self.BLOCK_SIZE - 1, y), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1), 2)
            pygame.draw.line(self.screen, darker_color, (x, y + self.BLOCK_SIZE - 1), (x + self.BLOCK_SIZE - 1, y + self.BLOCK_SIZE - 1), 2)


    def _draw_block_shape(self, shape, pos, color_idx, is_ghost=False):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(pos[0] + c, pos[1] + r, color_idx, is_ghost)

    def _render_ui(self):
        # --- Score ---
        score_text = self.font_title.render("SCORE", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(score_text, (30, 30))
        score_val = self.font_main.render(f"{int(self.score):06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_val, (30, 55))

        # --- Lines ---
        lines_text = self.font_title.render("LINES", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(lines_text, (30, 105))
        lines_val = self.font_main.render(f"{self.lines_cleared}/{self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_val, (30, 130))

        # --- Moves Left ---
        moves_text = self.font_title.render("MOVES", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(moves_text, (self.WIDTH - 150, 30))
        moves_val_color = self.COLOR_TEXT if self.moves_left > 20 else (255, 100, 100)
        moves_val = self.font_main.render(f"{self.moves_left:03d}", True, moves_val_color)
        self.screen.blit(moves_val, (self.WIDTH - 150, 55))

        # --- Next Block ---
        next_text = self.font_title.render("NEXT", True, self.COLOR_TEXT_ACCENT)
        self.screen.blit(next_text, (self.WIDTH - 150, 105))
        if self.next_block_idx is not None:
            next_shape = self.TETROMINOES[self.next_block_idx]
            shape_w = len(next_shape[0]) * self.BLOCK_SIZE
            shape_h = len(next_shape) * self.BLOCK_SIZE
            start_x = self.WIDTH - 150 + (120 - shape_w) / 2
            start_y = 135 + (60 - shape_h) / 2
            
            for r, row in enumerate(next_shape):
                for c, cell in enumerate(row):
                    if cell:
                        color = self.BLOCK_COLORS[self.next_block_idx]
                        pygame.draw.rect(self.screen, color, (start_x + c * self.BLOCK_SIZE, start_y + r * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE))

        # --- Game Over/Win Message ---
        if self.terminated:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.lines_cleared >= self.WIN_CONDITION_LINES:
                msg = "YOU WIN!"
                color = (100, 255, 100)
            else:
                msg = "GAME OVER"
                color = (255, 100, 100)

            end_text = pygame.font.SysFont("Consolas", 60, bold=True).render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # Example of how to run the environment
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Run headless
    
    env = GameEnv(render_mode="rgb_array")
    
    # --- Play with a random agent ---
    obs, info = env.reset(seed=42)
    print(f"Initial state: {info}")
    
    terminated = False
    total_reward = 0
    
    for i in range(500):
        if terminated:
            print(f"Game ended after {i+1} steps. Final Info: {info}")
            break
            
        action = env.action_space.sample() # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # To visualize, you would save the observation `obs` as an image
        if i % 50 == 0:
            print(f"Step {i}: Action={action}, Reward={reward:.2f}, Info={info}")

    print(f"Total reward after random play: {total_reward}")
    
    # --- To run interactively (requires removing the headless os.environ) ---
    #
    # del os.environ["SDL_VIDEODRIVER"]
    # pygame.display.set_caption("Pixel Fall")
    # env = GameEnv()
    # screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    # obs, info = env.reset()
    #
    # running = True
    # while running:
    #     action = [0, 0, 0] # Default action: no-op / soft drop
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False
    #
    #     keys = pygame.key.get_pressed()
    #     if keys[pygame.K_LEFT]: action[0] = 3
    #     elif keys[pygame.K_RIGHT]: action[0] = 4
    #     elif keys[pygame.K_UP]: action[0] = 1 # Rotate
    #     elif keys[pygame.K_DOWN]: action[0] = 2 # Soft drop
    #
    #     if keys[pygame.K_SPACE]: action[1] = 1 # Hard drop
    #
    #     obs, reward, terminated, _, info = env.step(action)
    #
    #     # Blit the observation from the env to the display screen
    #     surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    #     screen.blit(surf, (0, 0))
    #     pygame.display.flip()
    #
    #     if terminated:
    #         print(f"Game Over! Final Score: {info['score']}")
    #         pygame.time.wait(2000) # Wait 2 seconds
    #         obs, info = env.reset() # Restart
    #
    #     env.clock.tick(10) # Control game speed for human play
    #
    # env.close()