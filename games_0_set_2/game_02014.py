
# Generated: 2025-08-27T18:59:38.268651
# Source Brief: brief_02014.md
# Brief Index: 2014

        
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
        "Controls: ←→ to move, ↑ to rotate. Hold ↓ or SPACE for soft drop. Hold SHIFT for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, falling block puzzle. Clear lines to score points and win before the blocks stack to the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        
        # Playfield position
        self.GRID_X = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

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
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 16)
        
        # Colors
        self._define_colors()
        
        # Tetromino shapes and their rotations
        self._define_tetrominoes()

        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.fall_time = None
        self.fall_speed = None
        self.lines_cleared = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.line_clear_animation = []

        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Commented out for submission, but useful for testing

    def _define_colors(self):
        self.COLOR_BG = (25, 30, 45)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_GHOST = (255, 255, 255, 60)
        self.COLOR_FLASH = (255, 255, 255)
        
        self.TETROMINO_COLORS = [
            (0, 240, 240),   # I (Cyan)
            (240, 240, 0),   # O (Yellow)
            (160, 0, 240),   # T (Purple)
            (0, 240, 0),     # S (Green)
            (240, 0, 0),     # Z (Red)
            (0, 0, 240),     # J (Blue)
            (240, 160, 0),   # L (Orange)
        ]

    def _define_tetrominoes(self):
        # Shapes are defined by a list of (y, x) offsets from a central pivot
        self.TETROMINOES = {
            0: [[(0, -1), (0, 0), (0, 1), (0, 2)], [(-1, 0), (0, 0), (1, 0), (2, 0)]],  # I
            1: [[(0, 0), (0, 1), (1, 0), (1, 1)]],  # O
            2: [[(0, -1), (0, 0), (0, 1), (1, 0)], [(0, 1), (1, 0), (-1, 0), (0, 0)], [(0, -1), (0, 0), (0, 1), (-1, 0)], [(0,-1), (1, 0), (-1, 0), (0, 0)]],  # T
            3: [[(0, -1), (0, 0), (1, 0), (1, 1)], [(-1, 0), (0, 0), (0, -1), (1, -1)]], # S
            4: [[(0, 1), (0, 0), (1, 0), (1, -1)], [(-1, -1), (0, -1), (0, 0), (1, 0)]], # Z
            5: [[(0, -1), (0, 0), (0, 1), (1, -1)], [(-1, 0), (0, 0), (1, 0), (-1, 1)], [(0, -1), (0, 0), (0, 1), (-1, 1)], [(-1, 0), (0, 0), (1, 0), (1, -1)]], # J
            6: [[(0, -1), (0, 0), (0, 1), (1, 1)], [(-1, 0), (0, 0), (1, 0), (1, 1)], [(0, -1), (0, 0), (0, 1), (-1, -1)], [(-1, -1), (-1, 0), (0, 0), (1, 0)]], # L
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.fall_time = 0.0
        self.fall_speed = 1.0  # seconds per drop
        self.line_clear_animation = []

        self._new_piece()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.clock.tick(30) # Maintain 30 FPS for smooth visuals and timing
        
        reward = -0.01  # Small penalty per step to encourage efficiency

        # Handle player actions
        reward += self._handle_input(action)
        
        # Update piece falling
        if not self.game_over:
            reward += self._update_fall()

        # Check for termination conditions
        terminated = self.game_over or self.steps >= 1000
        if self.game_over:
            reward += -100  # Loss penalty
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0

        if shift_held:
            # Hard drop takes precedence
            reward += self._hard_drop()
            # SFX: Hard drop sound
        else:
            # Handle standard movements
            if movement == 1: self._rotate_piece() # Up -> Rotate
            elif movement == 3: self._move(-1) # Left
            elif movement == 4: self._move(1) # Right
            
            # Soft drop
            if movement == 2 or space_held: # Down or Space
                reward += self._soft_drop()
        
        return reward

    def _update_fall(self):
        self.fall_time += self.clock.get_time() / 1000.0
        reward = 0
        if self.fall_time >= self.fall_speed:
            self.fall_time = 0
            self.current_piece['y'] += 1
            if self._check_collision(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'], self.current_piece['y'])):
                self.current_piece['y'] -= 1
                reward += self._lock_piece()
                self._new_piece()
        return reward

    def _new_piece(self):
        shape_idx = self.np_random.integers(0, len(self.TETROMINOES))
        self.current_piece = {
            'shape_idx': shape_idx,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2,
            'y': 0,
            'color_idx': shape_idx,
        }
        if self._check_collision(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'], self.current_piece['y'])):
            self.game_over = True

    def _get_piece_coords(self, shape_idx, rotation, offset):
        shape = self.TETROMINOES[shape_idx][rotation % len(self.TETROMINOES[shape_idx])]
        coords = []
        for dy, dx in shape:
            coords.append((offset[0] + dx, offset[1] + dy))
        return coords

    def _check_collision(self, shape_idx, rotation, offset):
        for x, y in self._get_piece_coords(shape_idx, rotation, offset):
            if not (0 <= x < self.GRID_WIDTH and 0 <= y < self.GRID_HEIGHT):
                return True  # Wall collision
            if self.grid[y, x] > 0:
                return True  # Placed block collision
        return False

    def _move(self, dx):
        if not self._check_collision(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'] + dx, self.current_piece['y'])):
            self.current_piece['x'] += dx
            # SFX: Move sound

    def _rotate_piece(self):
        new_rotation = (self.current_piece['rotation'] + 1) % len(self.TETROMINOES[self.current_piece['shape_idx']])
        # Basic wall kick: try to nudge left/right if rotation fails
        for dx in [0, 1, -1, 2, -2]:
            if not self._check_collision(self.current_piece['shape_idx'], new_rotation, (self.current_piece['x'] + dx, self.current_piece['y'])):
                self.current_piece['rotation'] = new_rotation
                self.current_piece['x'] += dx
                # SFX: Rotate sound
                break

    def _soft_drop(self):
        if not self._check_collision(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'], self.current_piece['y'] + 1)):
            self.current_piece['y'] += 1
            self.fall_time = 0 # Reset gravity timer
            return 0.1 # Reward for dropping one row
        return 0

    def _hard_drop(self):
        rows_dropped = 0
        while not self._check_collision(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'], self.current_piece['y'] + 1)):
            self.current_piece['y'] += 1
            rows_dropped += 1
        
        lock_reward = self._lock_piece()
        self._new_piece()
        return lock_reward + (rows_dropped * 0.1)

    def _lock_piece(self):
        # Calculate placement reward before locking
        placement_reward = self._calculate_placement_reward()

        coords = self._get_piece_coords(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'], self.current_piece['y']))
        for x, y in coords:
            if 0 <= y < self.GRID_HEIGHT and 0 <= x < self.GRID_WIDTH:
                self.grid[y, x] = self.current_piece['color_idx'] + 1
        
        # Clear lines and get reward
        lines_cleared, line_reward = self._clear_lines()
        self.lines_cleared += lines_cleared
        self.score += int(line_reward * 10)
        
        # Update fall speed based on total lines cleared
        self.fall_speed = max(0.2, 1.0 - self.lines_cleared * 0.05)

        # Check for win condition
        if self.lines_cleared >= 10:
            self.game_over = True
            return placement_reward + line_reward + 100 # Win reward
        
        return placement_reward + line_reward

    def _calculate_placement_reward(self):
        coords = self._get_piece_coords(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'], self.current_piece['y']))
        
        is_safe = True
        has_overhang = False
        
        for x, y in coords:
            # Check for overhang
            if y + 1 < self.GRID_HEIGHT and self.grid[y + 1, x] == 0:
                has_overhang = True
            # Check if not safe
            if y + 1 >= self.GRID_HEIGHT or self.grid[y + 1, x] == 0:
                is_safe = False

        if has_overhang: return 2.0 # Risky play bonus
        if is_safe: return -0.2 # Safe play penalty
        return 0

    def _clear_lines(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y, :] > 0):
                lines_to_clear.append(y)
        
        if not lines_to_clear:
            return 0, 0

        # SFX: Line clear sound
        for y in lines_to_clear:
            self.grid[y, :] = 0
            self.line_clear_animation.append({'y': y, 'alpha': 255})
        
        # Shift rows down
        new_grid = np.zeros_like(self.grid)
        new_y = self.GRID_HEIGHT - 1
        for y in range(self.GRID_HEIGHT - 1, -1, -1):
            if y not in lines_to_clear:
                new_grid[new_y, :] = self.grid[y, :]
                new_y -= 1
        self.grid = new_grid

        num_cleared = len(lines_to_clear)
        # Bonus for clearing multiple lines at once (e.g., Tetris)
        reward = num_cleared * (1.0 + (num_cleared - 1) * 0.5)
        return num_cleared, reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._draw_grid_background()
        self._draw_placed_blocks()
        if not self.game_over:
            self._draw_ghost_piece()
            self._draw_current_piece()
        self._draw_line_clear_animation()

    def _draw_block(self, x, y, color_idx, surface=None, alpha=255):
        if surface is None:
            surface = self.screen
        
        color = self.TETROMINO_COLORS[color_idx]
        dark_color = tuple(c * 0.6 for c in color)
        
        px, py = self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE
        
        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        inner_rect = pygame.Rect(px + 2, py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4)

        if alpha < 255:
            target_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape_surf, (*dark_color, alpha), shape_surf.get_rect())
            pygame.draw.rect(shape_surf, (*color, alpha), inner_rect.move(-px, -py))
            surface.blit(shape_surf, target_rect)
        else:
            pygame.draw.rect(surface, dark_color, rect)
            pygame.draw.rect(surface, color, inner_rect)

    def _draw_grid_background(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = pygame.Rect(self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
    
    def _draw_placed_blocks(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] > 0:
                    color_idx = int(self.grid[y, x]) - 1
                    self._draw_block(x, y, color_idx)

    def _draw_current_piece(self):
        coords = self._get_piece_coords(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'], self.current_piece['y']))
        for x, y in coords:
            self._draw_block(x, y, self.current_piece['color_idx'])

    def _draw_ghost_piece(self):
        ghost_y = self.current_piece['y']
        while not self._check_collision(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'], ghost_y + 1)):
            ghost_y += 1
        
        coords = self._get_piece_coords(self.current_piece['shape_idx'], self.current_piece['rotation'], (self.current_piece['x'], ghost_y))
        
        for x, y in coords:
            px, py = self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE
            rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            
            # Create a semi-transparent surface
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            color = self.TETROMINO_COLORS[self.current_piece['color_idx']]
            s.fill((*color, 60))
            self.screen.blit(s, rect.topleft)

    def _draw_line_clear_animation(self):
        if not self.line_clear_animation:
            return
        
        next_animation_list = []
        for anim in self.line_clear_animation:
            y, alpha = anim['y'], anim['alpha']
            rect = pygame.Rect(self.GRID_X, self.GRID_Y + y * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
            s = pygame.Surface(rect.size, pygame.SRCALPHA)
            s.fill((*self.COLOR_FLASH, alpha))
            self.screen.blit(s, rect.topleft)
            
            anim['alpha'] -= 30 # Fade out speed
            if anim['alpha'] > 0:
                next_animation_list.append(anim)
        self.line_clear_animation = next_animation_list

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        
        # Lines
        lines_text = self.font_large.render(f"LINES: {self.lines_cleared} / 10", True, self.COLOR_UI_TEXT)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - lines_text.get_width() - 20, 20))

        # Game Over message
        if self.game_over:
            win = self.lines_cleared >= 10
            msg = "YOU WIN!" if win else "GAME OVER"
            color = (100, 255, 100) if win else (255, 100, 100)
            
            over_text = self.font_large.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            # Draw a backing for the text
            bg_rect = text_rect.inflate(20, 10)
            s = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, bg_rect)
            
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "game_over": self.game_over,
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    
    # Use a different screen for display to avoid conflicts with env's internal surface
    display_screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    pygame.display.set_caption("Falling Block Puzzle")
    
    terminated = False
    running = True
    
    while running:
        # Default action is no-op
        action = [0, 0, 0] # [movement, space, shift]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            
            # Map keys to actions
            # Note: This simple mapping doesn't handle discrete presses vs holds well for movement.
            # The agent's policy would decide this on a per-step basis.
            # For human play, we check once per frame.
            if keys[pygame.K_UP]: action[0] = 1
            elif keys[pygame.K_DOWN]: action[0] = 2
            elif keys[pygame.K_LEFT]: action[0] = 3
            elif keys[pygame.K_RIGHT]: action[0] = 4
            
            if keys[pygame.K_SPACE]: action[1] = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        env.clock.tick(30) # Control human play speed

    env.close()