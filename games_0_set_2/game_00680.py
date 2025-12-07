
# Generated: 2025-08-27T14:26:09.688863
# Source Brief: brief_00680.md
# Brief Index: 680

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, Shift to rotate counter-clockwise. "
        "↓ to soft drop, Space to hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Position falling shapes to complete lines for points "
        "before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Game Constants ---
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_BLOCK_SIZE = 18
    GRID_LINE_WIDTH = 1

    # Centering the grid on the screen
    GRID_X_OFFSET = (SCREEN_WIDTH - GRID_WIDTH * GRID_BLOCK_SIZE) // 2
    GRID_Y_OFFSET = (SCREEN_HEIGHT - GRID_HEIGHT * GRID_BLOCK_SIZE) // 2

    # Colors
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_GHOST = (255, 255, 255, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_WARN = (255, 50, 50)
    
    SHAPE_COLORS = [
        (0, 255, 255),  # I (Cyan)
        (0, 0, 255),    # J (Blue)
        (255, 165, 0),  # L (Orange)
        (255, 255, 0),  # O (Yellow)
        (0, 255, 0),    # S (Green)
        (128, 0, 128),  # T (Purple)
        (255, 0, 0),    # Z (Red)
    ]

    # Tetromino shapes and their rotations
    SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 0, 0], [1, 1, 1]],  # J
        [[0, 0, 1], [1, 1, 1]],  # L
        [[1, 1], [1, 1]],  # O
        [[0, 1, 1], [1, 1, 0]],  # S
        [[0, 1, 0], [1, 1, 1]],  # T
        [[1, 1, 0], [0, 1, 1]],  # Z
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
        self.font_main = pygame.font.SysFont("monospace", 24, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 16)
        
        self.grid = None
        self.current_piece = None
        self.next_piece_id = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0
        self.fall_progress = 0
        self.line_clear_animation_timer = 0
        self.last_cleared_rows = []
        
        self._precompute_rotations()
        
        self.reset()
        
        # self.validate_implementation() # Optional: Call for self-check

    def _precompute_rotations(self):
        self.ROTATED_SHAPES = []
        for shape in self.SHAPES:
            rotations = []
            current_shape = np.array(shape)
            for _ in range(4):
                rotations.append(current_shape.tolist())
                current_shape = np.rot90(current_shape)
            self.ROTATED_SHAPES.append(rotations)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0.5
        self.fall_progress = 0.0
        self.line_clear_animation_timer = 0
        self.last_cleared_rows = []
        
        self.piece_bag = list(range(len(self.SHAPES)))
        random.shuffle(self.piece_bag)
        
        self._spawn_new_piece()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = -0.01  # Per-step penalty
        self.steps += 1
        
        # Handle line clear animation pause
        if self.line_clear_animation_timer > 0:
            self.line_clear_animation_timer -= 1
            if self.line_clear_animation_timer == 0:
                self._perform_line_clear()
            # Skip game logic during pause, just render
            return self._get_observation(), reward, self.game_over, False, self._get_info()

        # --- Action Handling ---
        movement = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        if space_held:
            # Hard drop
            drop_distance = 0
            while self._is_valid_position(self.current_piece, (0, 1)):
                self.current_piece["y"] += 1
                drop_distance += 1
            # Hard drop is a terminal move for this piece
            reward += self._lock_piece()
        else:
            # Handle lateral movement and rotation
            if movement == 3:  # Left
                self._move(-1, 0)
            elif movement == 4:  # Right
                self._move(1, 0)
            
            if movement == 1:  # Rotate Clockwise
                self._rotate(1)
            elif shift_held:   # Rotate Counter-Clockwise (Shift)
                self._rotate(-1)
            
            if movement == 2: # Soft drop
                if self._move(0, 1):
                    self.fall_progress = 0 # Reset gravity progress after manual drop
                    # sound: soft_drop_tick

            # --- Gravity ---
            self.fall_progress += self.fall_speed
            if self.fall_progress >= 1.0:
                moves_due_to_gravity = int(self.fall_progress)
                for _ in range(moves_due_to_gravity):
                    if not self._move(0, 1):
                        # Piece hit something, lock it
                        reward += self._lock_piece()
                        break # Exit gravity loop if locked
                self.fall_progress %= 1.0
        
        # --- Termination Checks ---
        if self.score >= 1000 and not self.game_over:
            self.game_over = True
            reward += 100 # Win bonus
        
        if self.steps >= 10000 and not self.game_over:
            self.game_over = True # Max steps reached
        
        terminated = self.game_over
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _move(self, dx, dy):
        if self._is_valid_position(self.current_piece, (dx, dy)):
            self.current_piece["x"] += dx
            self.current_piece["y"] += dy
            return True
        return False

    def _rotate(self, direction):
        old_rotation = self.current_piece["rotation"]
        new_rotation = (old_rotation + direction) % 4
        self.current_piece["rotation"] = new_rotation
        
        if not self._is_valid_position(self.current_piece):
            # Wall kick logic
            for dx in [-1, 1, -2, 2]: # Simple wall kick
                if self._is_valid_position(self.current_piece, (dx, 0)):
                    self.current_piece["x"] += dx
                    # sound: rotate_success
                    return
            self.current_piece["rotation"] = old_rotation # Revert if no valid position found
            # sound: rotate_fail
        else:
            # sound: rotate_success
            pass

    def _is_valid_position(self, piece, offset=(0, 0)):
        shape = self.ROTATED_SHAPES[piece["id"]][piece["rotation"]]
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    grid_x = piece["x"] + c_idx + offset[0]
                    grid_y = piece["y"] + r_idx + offset[1]
                    
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False # Out of bounds
                    if self.grid[grid_y, grid_x] != 0:
                        return False # Collision with existing piece
        return True

    def _lock_piece(self):
        # sound: piece_lock
        placement_reward = 0
        shape = self.ROTATED_SHAPES[self.current_piece["id"]][self.current_piece["rotation"]]
        max_y = 0
        
        for r_idx, row in enumerate(shape):
            for c_idx, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece["x"] + c_idx
                    grid_y = self.current_piece["y"] + r_idx
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[grid_y, grid_x] = self.current_piece["id"] + 1
                        max_y = max(max_y, grid_y)

        # Placement rewards
        if max_y < 2: # Top 2 rows
            placement_reward -= 0.2
        elif max_y >= self.GRID_HEIGHT - 2: # Bottom 2 rows
            placement_reward += 0.1

        lines_cleared = self._check_for_line_clears()
        
        if lines_cleared > 0:
            # Rewards for clearing lines
            line_rewards = {1: 1, 2: 3, 3: 7, 4: 15}
            placement_reward += line_rewards.get(lines_cleared, 0)
            self.line_clear_animation_timer = 5 # Set pause duration (in frames)
            # sound: line_clear
        else:
            self._spawn_new_piece()

        return placement_reward

    def _check_for_line_clears(self):
        self.last_cleared_rows = []
        for r in range(self.GRID_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                self.last_cleared_rows.append(r)
        
        if self.last_cleared_rows:
            new_score = self.score + len(self.last_cleared_rows) * 100
            # Update difficulty
            if self.score // 200 != new_score // 200:
                self.fall_speed += 0.02
            self.score = new_score
        
        return len(self.last_cleared_rows)
    
    def _perform_line_clear(self):
        if not self.last_cleared_rows:
            return

        # Remove rows from bottom up
        for row_idx in sorted(self.last_cleared_rows, reverse=True):
            self.grid = np.delete(self.grid, row_idx, axis=0)
        
        # Add new empty rows at the top
        new_rows = np.zeros((len(self.last_cleared_rows), self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack((new_rows, self.grid))
        
        self.last_cleared_rows = []
        self._spawn_new_piece()

    def _spawn_new_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.SHAPES)))
            random.shuffle(self.piece_bag)
        
        piece_id = self.piece_bag.pop()
        
        self.current_piece = {
            "id": piece_id,
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 1,
            "y": 0,
        }
        
        if not self._is_valid_position(self.current_piece):
            self.game_over = True
            # sound: game_over
    
    def _get_ghost_position(self):
        if not self.current_piece or self.game_over:
            return None
        ghost_piece = self.current_piece.copy()
        while self._is_valid_position(ghost_piece, (0, 1)):
            ghost_piece["y"] += 1
        return ghost_piece

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y_OFFSET + r * self.GRID_BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X_OFFSET, y), (self.GRID_X_OFFSET + self.GRID_WIDTH * self.GRID_BLOCK_SIZE, y), self.GRID_LINE_WIDTH)
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_X_OFFSET + c * self.GRID_BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y_OFFSET), (x, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.GRID_BLOCK_SIZE), self.GRID_LINE_WIDTH)

        # Draw locked pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_block(c, r, self.SHAPE_COLORS[self.grid[r, c] - 1])

        # Draw ghost piece
        ghost = self._get_ghost_position()
        if ghost:
            shape = self.ROTATED_SHAPES[ghost["id"]][ghost["rotation"]]
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_block(ghost["x"] + c_idx, ghost["y"] + r_idx, self.COLOR_GHOST, is_ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            shape = self.ROTATED_SHAPES[self.current_piece["id"]][self.current_piece["rotation"]]
            color = self.SHAPE_COLORS[self.current_piece["id"]]
            for r_idx, row in enumerate(shape):
                for c_idx, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece["x"] + c_idx, self.current_piece["y"] + r_idx, color)

        # Draw line clear animation
        if self.line_clear_animation_timer > 0:
            flash_color = (255, 255, 255, 150)
            for row_idx in self.last_cleared_rows:
                rect = pygame.Rect(
                    self.GRID_X_OFFSET,
                    self.GRID_Y_OFFSET + row_idx * self.GRID_BLOCK_SIZE,
                    self.GRID_WIDTH * self.GRID_BLOCK_SIZE,
                    self.GRID_BLOCK_SIZE
                )
                flash_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
                flash_surface.fill(flash_color)
                self.screen.blit(flash_surface, rect.topleft)

    def _draw_block(self, grid_x, grid_y, color, is_ghost=False):
        screen_x = self.GRID_X_OFFSET + grid_x * self.GRID_BLOCK_SIZE
        screen_y = self.GRID_Y_OFFSET + grid_y * self.GRID_BLOCK_SIZE
        rect = pygame.Rect(screen_x, screen_y, self.GRID_BLOCK_SIZE, self.GRID_BLOCK_SIZE)

        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2) # Draw outline for ghost
        else:
            # Main block color
            pygame.draw.rect(self.screen, color, rect)
            # 3D effect
            highlight_color = tuple(min(255, c + 50) for c in color)
            shadow_color = tuple(max(0, c - 50) for c in color)
            
            # Draw highlight on top and left
            pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.topright)
            pygame.draw.line(self.screen, highlight_color, rect.topleft, rect.bottomleft)
            # Draw shadow on bottom and right
            pygame.draw.line(self.screen, shadow_color, rect.bottomright, rect.topright)
            pygame.draw.line(self.screen, shadow_color, rect.bottomright, rect.bottomleft)

    def _render_ui(self):
        # Score display
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Warning flash for high stack
        if np.any(self.grid[:2, :] != 0):
            pulse = (math.sin(self.steps * 0.2) + 1) / 2 # 0 to 1 pulse
            alpha = int(50 + pulse * 100)
            warn_surface = pygame.Surface((self.GRID_WIDTH * self.GRID_BLOCK_SIZE, 2 * self.GRID_BLOCK_SIZE), pygame.SRCALPHA)
            warn_surface.fill((*self.COLOR_WARN, alpha))
            self.screen.blit(warn_surface, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET))

        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "YOU WIN!" if self.score >= 1000 else "GAME OVER"
            end_text = self.font_main.render(end_text_str, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "fall_speed": self.fall_speed,
        }
        
    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
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

if __name__ == "__main__":
    # --- Manual Play Loop ---
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Gymnasium Game")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    print("\n" + "="*30)
    print("MANUAL PLAY MODE")
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while running:
        movement = 0  # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                print("Resetting environment.")
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
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode Finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()