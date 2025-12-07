
# Generated: 2025-08-27T16:46:08.538208
# Source Brief: brief_01316.md
# Brief Index: 1316

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
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
        "Controls: ←→ to move, ↓ for soft drop, ↑ to rotate clockwise. "
        "Hold Shift to rotate counter-clockwise. Press Space for hard drop."
    )

    game_description = (
        "A fast-paced, grid-based puzzle game. Manipulate falling shapes to clear lines, "
        "but beware! The speed increases as you score higher. The game ends if the stack "
        "reaches the top."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.BLOCK_SIZE = 18
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2
        self.GRID_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2

        # Colors
        self.COLOR_BG = (32, 32, 32)
        self.COLOR_GRID = (64, 64, 64)
        self.COLOR_GHOST = (128, 128, 128, 100)
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_CLEAR_ANIM = (255, 255, 255)
        
        # Shapes and their colors. Using 7 standard tetrominoes and 3 pentominoes to meet the '10 shapes' requirement.
        self._define_shapes_and_colors()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables (will be initialized in reset)
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.fall_speed = 0
        self.lines_cleared_anim = []
        
        # Action handling state
        self.last_up_pressed = False
        self.last_space_held = False
        self.last_shift_held = False
        
        # RNG
        self.rng = None

        self.reset()
        self.validate_implementation()

    def _define_shapes_and_colors(self):
        # Shape matrices define the block layout
        self.SHAPES = [
            [[1, 1, 1, 1]],  # I
            [[1, 1], [1, 1]],  # O
            [[0, 1, 0], [1, 1, 1]],  # T
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 0, 1], [1, 1, 1]],  # L
            [[1, 1, 0], [0, 1, 1]],  # Z
            [[0, 1, 1], [1, 1, 0]],  # S
            [[1, 0, 1], [1, 1, 1]],  # U (Pentomino)
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # + (Pentomino)
            [[1, 1, 0], [0, 1, 0], [0, 1, 1]],  # F (Pentomino)
        ]
        self.SHAPE_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 255, 0),  # Yellow
            (128, 0, 128),  # Purple
            (0, 0, 255),    # Blue
            (255, 165, 0),  # Orange
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (255, 105, 180),# Hot Pink
            (192, 192, 192),# Silver
            (240, 230, 140),# Khaki
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_timer = 0
        self.fall_speed = 1.0  # Time in seconds to fall one block
        self.lines_cleared_anim = []
        
        self.last_up_pressed = False
        self.last_space_held = False
        self.last_shift_held = False

        self.next_piece = self._new_piece()
        self._spawn_new_piece()

        return self._get_observation(), self._get_info()

    def _new_piece(self):
        shape_idx = self.rng.integers(0, len(self.SHAPES))
        shape = self.SHAPES[shape_idx]
        return {
            'shape_idx': shape_idx,
            'shape': shape,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'color': self.SHAPE_COLORS[shape_idx]
        }

    def _spawn_new_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()
        self.fall_timer = 0
        if not self._is_valid_position(self.current_piece):
            self.game_over = True

    def step(self, action):
        if self.game_over:
            return self._get_observation(), -10.0, True, False, self._get_info()

        self.steps += 1
        reward = 0.0
        
        # Handle line clear animation
        self._update_animations()
        
        # Handle player input
        soft_dropped, hard_dropped = self._handle_input(action)
        if soft_dropped:
            reward += 0.1

        # If a hard drop occurred, the piece is locked and a new one is spawned.
        # The reward for line clears is handled inside _handle_input.
        if hard_dropped:
            reward += self._lock_and_clear()
        else:
            # Automatic fall
            self.fall_timer += 1 / 30.0  # Assuming 30 FPS
            if self.fall_timer >= self.fall_speed:
                self.fall_timer = 0
                if self._is_valid_position(self.current_piece, offset=(0, 1)):
                    self.current_piece['y'] += 1
                else:
                    reward += self._lock_and_clear()
        
        terminated = self._check_termination()
        if terminated:
            if self.score >= 1000:
                reward = 100.0 # Win reward
            else:
                reward = -10.0 # Lose reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        soft_dropped = False
        hard_dropped = False

        # Rising edge detection for one-shot actions
        up_pressed = movement == 1
        rotate_cw = up_pressed and not self.last_up_pressed
        self.last_up_pressed = up_pressed

        rotate_ccw = shift_held and not self.last_shift_held
        self.last_shift_held = shift_held

        hard_drop = space_held and not self.last_space_held
        self.last_space_held = space_held

        if hard_drop:
            # Sound: Hard drop
            while self._is_valid_position(self.current_piece, offset=(0, 1)):
                self.current_piece['y'] += 1
            hard_dropped = True
            return soft_dropped, hard_dropped

        if rotate_cw:
            # Sound: Rotate
            self._rotate_piece(self.current_piece)
        if rotate_ccw:
            # Sound: Rotate
            self._rotate_piece(self.current_piece, clockwise=False)

        if movement == 3:  # Left
            if self._is_valid_position(self.current_piece, offset=(-1, 0)):
                self.current_piece['x'] -= 1
        elif movement == 4:  # Right
            if self._is_valid_position(self.current_piece, offset=(1, 0)):
                self.current_piece['x'] += 1
        
        if movement == 2:  # Down (Soft Drop)
            if self._is_valid_position(self.current_piece, offset=(0, 1)):
                self.current_piece['y'] += 1
                self.fall_timer = 0 # Reset auto-fall timer
                soft_dropped = True

        return soft_dropped, hard_dropped

    def _lock_and_clear(self):
        # Sound: Piece lock
        self._place_piece_on_grid()
        lines_cleared = self._clear_lines()
        
        reward = 0
        if lines_cleared > 0:
            # Sound: Line clear
            line_rewards = {1: 10, 2: 30, 3: 50, 4: 100}
            reward = line_rewards.get(lines_cleared, 100)
            
            old_score_tier = self.score // 100
            self.score += reward
            new_score_tier = self.score // 100
            
            # Increase speed for every 100 points gained
            if new_score_tier > old_score_tier:
                speed_increase = (new_score_tier - old_score_tier) * 0.05
                self.fall_speed = max(0.1, self.fall_speed - speed_increase)

        self._spawn_new_piece()
        return float(reward)

    def _rotate_piece(self, piece, clockwise=True):
        original_shape = piece['shape']
        if clockwise:
            new_shape = list(zip(*piece['shape'][::-1]))
        else:
            new_shape = list(zip(*piece['shape']))[::-1]
        
        piece['shape'] = new_shape
        if not self._is_valid_position(piece):
            # Wall kick logic (simplified: try moving left/right)
            if self._is_valid_position(piece, offset=(-1, 0)):
                piece['x'] -= 1
            elif self._is_valid_position(piece, offset=(1, 0)):
                piece['x'] += 1
            elif self._is_valid_position(piece, offset=(-2, 0)): # For 'I' piece
                piece['x'] -= 2
            elif self._is_valid_position(piece, offset=(2, 0)):
                piece['x'] += 2
            else: # Revert if no valid position found
                piece['shape'] = original_shape

    def _is_valid_position(self, piece, offset=(0, 0)):
        off_x, off_y = offset
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    x = piece['x'] + c + off_x
                    y = piece['y'] + r + off_y
                    if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT and self.grid[y][x] == 0):
                        return False
        return True

    def _place_piece_on_grid(self):
        for r, row in enumerate(self.current_piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    x = self.current_piece['x'] + c
                    y = self.current_piece['y'] + r
                    if 0 <= y < self.GRID_HEIGHT:
                        self.grid[y][x] = self.current_piece['shape_idx'] + 1

    def _clear_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.grid) if np.all(row)]
        if not lines_to_clear:
            return 0
        
        for r in lines_to_clear:
            self.lines_cleared_anim.append({'y': r, 'timer': 5}) # 5 frames animation
            self.grid = np.delete(self.grid, r, axis=0)
        
        new_rows = np.zeros((len(lines_to_clear), self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack([new_rows, self.grid])
        
        return len(lines_to_clear)

    def _update_animations(self):
        self.lines_cleared_anim = [anim for anim in self.lines_cleared_anim if anim['timer'] > 0]
        for anim in self.lines_cleared_anim:
            anim['timer'] -= 1

    def _check_termination(self):
        return self.game_over or self.score >= 1000 or self.steps >= 10000

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid lines
        for r in range(self.GRID_HEIGHT + 1):
            y = self.GRID_Y + r * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.GRID_X, y), (self.GRID_X + self.GRID_WIDTH * self.BLOCK_SIZE, y))
        for c in range(self.GRID_WIDTH + 1):
            x = self.GRID_X + c * self.BLOCK_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, self.GRID_Y), (x, self.GRID_Y + self.GRID_HEIGHT * self.BLOCK_SIZE))

        # Draw locked pieces
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r][c] > 0:
                    self._draw_block(c, r, self.SHAPE_COLORS[int(self.grid[r][c]) - 1])

        # Draw ghost piece
        if not self.game_over:
            ghost = self.current_piece.copy()
            while self._is_valid_position(ghost, offset=(0, 1)):
                ghost['y'] += 1
            self._draw_piece(ghost, ghost=True)

        # Draw falling piece
        if not self.game_over:
            self._draw_piece(self.current_piece)
            
        # Draw line clear animation
        for anim in self.lines_cleared_anim:
            y = self.GRID_Y + anim['y'] * self.BLOCK_SIZE
            rect = pygame.Rect(self.GRID_X, y, self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE)
            pygame.draw.rect(self.screen, self.COLOR_CLEAR_ANIM, rect)

    def _draw_piece(self, piece, ghost=False):
        for r, row in enumerate(piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(piece['x'] + c, piece['y'] + r, piece['color'], ghost)

    def _draw_block(self, x, y, color, ghost=False):
        rect = pygame.Rect(
            self.GRID_X + x * self.BLOCK_SIZE,
            self.GRID_Y + y * self.BLOCK_SIZE,
            self.BLOCK_SIZE, self.BLOCK_SIZE
        )
        if ghost:
            s = pygame.Surface((self.BLOCK_SIZE, self.BLOCK_SIZE), pygame.SRCALPHA)
            s.fill((*color, 80))
            self.screen.blit(s, rect.topleft)
            pygame.gfxdraw.rectangle(self.screen, rect, (*color, 120))
        else:
            pygame.draw.rect(self.screen, color, rect)
            border_color = tuple(max(0, val - 50) for val in color)
            pygame.draw.rect(self.screen, border_color, rect, 2)

    def _render_ui(self):
        # Score display
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Next piece preview
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 130, 20))
        
        preview_box = pygame.Rect(self.SCREEN_WIDTH - 150, 50, 120, 100)
        pygame.draw.rect(self.screen, self.COLOR_GRID, preview_box, 2, 5)

        if self.next_piece:
            shape = self.next_piece['shape']
            shape_w = len(shape[0]) * self.BLOCK_SIZE
            shape_h = len(shape) * self.BLOCK_SIZE
            start_x = preview_box.centerx - shape_w // 2
            start_y = preview_box.centery - shape_h // 2
            
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        rect = pygame.Rect(
                            start_x + c * self.BLOCK_SIZE,
                            start_y + r * self.BLOCK_SIZE,
                            self.BLOCK_SIZE, self.BLOCK_SIZE
                        )
                        pygame.draw.rect(self.screen, self.next_piece['color'], rect)
                        border_color = tuple(max(0, val - 50) for val in self.next_piece['color'])
                        pygame.draw.rect(self.screen, border_color, rect, 2)
                        
        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            win_or_lose_text = "YOU WIN!" if self.score >= 1000 else "GAME OVER"
            
            end_text = self.font_large.render(win_or_lose_text, True, self.COLOR_WHITE)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

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
    # This block allows you to play the game with keyboard controls
    # Requires pygame to be installed with display support.
    # To run in a headless environment, this block should be removed or commented out.
    try:
        import os
        # Set a non-dummy driver if you want to see the window
        if os.environ.get("SDL_VIDEODRIVER", "") == "dummy":
            print("Running in headless mode. No display will be shown.")
            # Run a few steps to ensure it doesn't crash
            env = GameEnv()
            for _ in range(100):
                action = env.action_space.sample()
                env.step(action)
            env.close()
            exit()

        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        running = True
        terminated = False
        
        # Map pygame keys to gymnasium actions
        key_to_action = {
            pygame.K_LEFT: 3,
            pygame.K_RIGHT: 4,
            pygame.K_UP: 1,
            pygame.K_DOWN: 2,
        }

        # Create a window to display the game
        pygame.display.set_caption("Gymnasium Game")
        display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))

        while running:
            # Construct the action from keyboard state
            movement_action = 0 # No-op
            space_action = 0
            shift_action = 0

            keys = pygame.key.get_pressed()
            for key, move_val in key_to_action.items():
                if keys[key]:
                    movement_action = move_val
                    break # Prioritize one movement key
            
            if keys[pygame.K_SPACE]:
                space_action = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                shift_action = 1

            action = [movement_action, space_action, shift_action]
            
            # Process pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    print("Resetting game.")
                    obs, info = env.reset()
                    terminated = False

            if not terminated:
                obs, reward, terminated, truncated, info = env.step(action)
                if reward != 0:
                    print(f"Step: {info['steps']}, Score: {info['score']}, Reward: {reward:.2f}")
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}")

            # Update the display
            # The observation is (H, W, C), but pygame blit needs a surface
            # So we get the surface directly from the env
            display_screen.blit(env.screen, (0, 0))
            pygame.display.flip()

            # Cap the frame rate
            env.clock.tick(30)

        pygame.quit()
    except ImportError:
        print("Pygame is not installed or display is not available. Cannot run interactive demo.")
    except pygame.error as e:
        print(f"Pygame error: {e}. Could not initialize display. Ensure you are not in a headless environment.")