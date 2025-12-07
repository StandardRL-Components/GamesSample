
# Generated: 2025-08-28T00:43:18.060190
# Source Brief: brief_03882.md
# Brief Index: 3882

        
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

    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Hold shift for soft drop, press space for hard drop."
    )

    game_description = (
        "A fast-paced, falling block puzzle. Clear lines by filling rows. Clear 10 lines to win, but don't let the stack reach the top!"
    )

    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    PLAYFIELD_WIDTH, PLAYFIELD_HEIGHT = 10, 20
    BLOCK_SIZE = 18
    
    GRID_X = (SCREEN_WIDTH - PLAYFIELD_WIDTH * BLOCK_SIZE) // 2
    GRID_Y = (SCREEN_HEIGHT - PLAYFIELD_HEIGHT * BLOCK_SIZE) // 2

    MAX_STEPS = 2500 # Increased to allow more time for a 10-line clear
    LINES_TO_WIN = 10
    
    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_GHOST = (255, 255, 255, 50)
    
    # Tetromino shapes and colors
    TETROMINOES = {
        'I': {'shape': [[1, 0], [1, 1], [1, 2], [1, 3]], 'color_idx': 1},
        'O': {'shape': [[0, 1], [0, 2], [1, 1], [1, 2]], 'color_idx': 2},
        'T': {'shape': [[0, 1], [1, 0], [1, 1], [1, 2]], 'color_idx': 3},
        'L': {'shape': [[0, 2], [1, 0], [1, 1], [1, 2]], 'color_idx': 4},
        'J': {'shape': [[0, 0], [1, 0], [1, 1], [1, 2]], 'color_idx': 5},
        'S': {'shape': [[0, 1], [0, 2], [1, 0], [1, 1]], 'color_idx': 6},
        'Z': {'shape': [[0, 0], [0, 1], [1, 1], [1, 2]], 'color_idx': 7},
    }
    
    BLOCK_COLORS = [
        (0, 0, 0),          # 0: Empty
        (0, 240, 240),      # 1: I (Cyan)
        (240, 240, 0),      # 2: O (Yellow)
        (160, 0, 240),      # 3: T (Purple)
        (240, 160, 0),      # 4: L (Orange)
        (0, 0, 240),        # 5: J (Blue)
        (0, 240, 0),        # 6: S (Green)
        (240, 0, 0),        # 7: Z (Red)
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
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)

        # Generate rotated shapes once
        self._rotated_tetrominoes = self._generate_rotations()
        
        self.reset()
        
        self.validate_implementation()

    def _generate_rotations(self):
        rotated = {}
        for name, data in self.TETROMINOES.items():
            shapes = [data['shape']]
            current_shape = data['shape']
            for _ in range(3):
                # Rotate 90 degrees clockwise: (x, y) -> (y, -x)
                # For grid coordinates (row, col), pivot around a point (e.g., 1.5, 1.5)
                # (r, c) -> (c, -r) relative to pivot
                pivot = (1.5, 1.5)
                new_shape = []
                for r, c in current_shape:
                    r_rel, c_rel = r - pivot[0], c - pivot[1]
                    new_c, new_r = -r_rel, c_rel
                    new_shape.append([new_r + pivot[0], new_c + pivot[1]])

                # Normalize to have min row/col at 0
                min_r = min(r for r, c in new_shape)
                min_c = min(c for r, c in new_shape)
                normalized_shape = sorted([[int(r - min_r), int(c - min_c)] for r, c in new_shape])
                
                if normalized_shape not in shapes:
                    shapes.append(normalized_shape)
                current_shape = normalized_shape
            rotated[name] = {'shapes': shapes, 'color_idx': data['color_idx']}
        return rotated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        
        self.bag = []
        self._fill_bag()
        
        self.current_piece = None
        self.next_piece_name = self._pull_from_bag()
        self._spawn_piece()
        
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.fall_progress = 0.0
        self.fall_speed = 0.5 # units per frame
        
        self.last_space_held = False
        self.action_cooldowns = {'move': 0, 'rotate': 0}
        
        self.line_clear_animation = None
        
        return self._get_observation(), self._get_info()

    def _fill_bag(self):
        self.bag = list(self.TETROMINOES.keys())
        self.np_random.shuffle(self.bag)

    def _pull_from_bag(self):
        if not self.bag:
            self._fill_bag()
        return self.bag.pop()

    def _spawn_piece(self):
        self.current_piece = {
            'name': self.next_piece_name,
            'rotation': 0,
            'x': self.PLAYFIELD_WIDTH // 2 - 2,
            'y': 0,
        }
        self.next_piece_name = self._pull_from_bag()
        
        if not self._is_valid_position(self.current_piece):
            self.game_over = True

    def step(self, action):
        reward = -0.01  # Small penalty for time passing
        self.steps += 1
        
        # Handle line clear animation delay
        if self.line_clear_animation:
            self.line_clear_animation['timer'] -= 1
            if self.line_clear_animation['timer'] <= 0:
                self._execute_line_clear()
                self.line_clear_animation = None
            # Return early, pausing gameplay during animation
            obs = self._get_observation()
            return obs, 0, self.game_over, False, self._get_info()

        if self.game_over:
            reward = -100
            obs = self._get_observation()
            return obs, reward, True, False, self._get_info()

        # Unpack actions
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        # --- Handle player input ---
        # Cooldowns to prevent excessively fast inputs in auto-advance mode
        for key in self.action_cooldowns:
            if self.action_cooldowns[key] > 0:
                self.action_cooldowns[key] -= 1

        # Movement
        if self.action_cooldowns['move'] == 0:
            if movement == 3: # Left
                self.current_piece['x'] -= 1
                if not self._is_valid_position(self.current_piece):
                    self.current_piece['x'] += 1
                self.action_cooldowns['move'] = 3 # 3-frame cooldown
            elif movement == 4: # Right
                self.current_piece['x'] += 1
                if not self._is_valid_position(self.current_piece):
                    self.current_piece['x'] -= 1
                self.action_cooldowns['move'] = 3

        # Rotation
        if self.action_cooldowns['rotate'] == 0:
            if movement == 1: # Up -> Rotate Right
                self._rotate_piece(1)
                self.action_cooldowns['rotate'] = 5
            elif movement == 2: # Down -> Rotate Left
                self._rotate_piece(-1)
                self.action_cooldowns['rotate'] = 5

        # Hard drop
        if space_held and not self.last_space_held:
            # Sound: Hard Drop
            while self._is_valid_position(self.current_piece, dy=1):
                self.current_piece['y'] += 1
            self._place_piece()
            reward += self._check_and_process_lines()
            self._spawn_piece()
            self.fall_progress = 0
        else:
            # --- Apply gravity ---
            current_fall_speed = self.fall_speed * 5.0 if shift_held else self.fall_speed
            self.fall_progress += current_fall_speed
            
            if self.fall_progress >= 1.0:
                moves = int(self.fall_progress)
                self.fall_progress -= moves
                for _ in range(moves):
                    if self._is_valid_position(self.current_piece, dy=1):
                        self.current_piece['y'] += 1
                    else:
                        old_height = self._get_stack_height()
                        self._place_piece()
                        new_height = self._get_stack_height()
                        
                        # Reward for height change
                        if new_height > old_height:
                            reward -= 0.2
                        
                        reward += self._check_and_process_lines()
                        self._spawn_piece()
                        break # Exit gravity loop after placing piece

        self.last_space_held = space_held
        
        # --- Termination conditions ---
        terminated = self.game_over or self.lines_cleared >= self.LINES_TO_WIN or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            if self.lines_cleared >= self.LINES_TO_WIN:
                reward += 100 # Win bonus
            # No penalty for max steps timeout
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _rotate_piece(self, direction):
        if self.current_piece['name'] == 'O': return # O doesn't rotate
        
        original_rotation = self.current_piece['rotation']
        num_rotations = len(self._rotated_tetrominoes[self.current_piece['name']]['shapes'])
        self.current_piece['rotation'] = (original_rotation + direction) % num_rotations
        
        # Wall kick implementation (SRS-like)
        for dx, dy in [(0,0), (-1,0), (1,0), (-2,0), (2,0), (0,-1), (0,-2)]:
            if self._is_valid_position(self.current_piece, dx=dx, dy=dy):
                self.current_piece['x'] += dx
                self.current_piece['y'] += dy
                # Sound: Rotate
                return
        
        # If no valid rotation found, revert
        self.current_piece['rotation'] = original_rotation

    def _get_piece_coords(self, piece, dx=0, dy=0, rotation_offset=0):
        shape_data = self._rotated_tetrominoes[piece['name']]
        num_rotations = len(shape_data['shapes'])
        rot_idx = (piece['rotation'] + rotation_offset) % num_rotations
        shape = shape_data['shapes'][rot_idx]
        
        coords = []
        for r, c in shape:
            coords.append((piece['y'] + r + dy, piece['x'] + c + dx))
        return coords

    def _is_valid_position(self, piece, dx=0, dy=0):
        coords = self._get_piece_coords(piece, dx, dy)
        for r, c in coords:
            if not (0 <= c < self.PLAYFIELD_WIDTH and 0 <= r < self.PLAYFIELD_HEIGHT):
                return False
            if r >= 0 and self.grid[r, c] != 0:
                return False
        return True

    def _place_piece(self):
        # Sound: Place piece
        shape_data = self._rotated_tetrominoes[self.current_piece['name']]
        color_idx = shape_data['color_idx']
        coords = self._get_piece_coords(self.current_piece)
        for r, c in coords:
            if r >= 0:
                self.grid[r, c] = color_idx
        self.current_piece = None

    def _check_and_process_lines(self):
        full_lines = []
        for r in range(self.PLAYFIELD_HEIGHT):
            if np.all(self.grid[r, :] != 0):
                full_lines.append(r)
        
        if full_lines:
            # Sound: Line clear
            self.score += len(full_lines)
            self.lines_cleared += len(full_lines)
            
            # Start animation
            self.line_clear_animation = {'rows': full_lines, 'timer': 8}
            
            # Update difficulty
            self.fall_speed = 0.5 + (self.lines_cleared // 2) * 0.05

            # Calculate reward
            reward_map = {1: 1, 2: 3, 3: 7, 4: 15}
            return reward_map.get(len(full_lines), 0)
        return 0

    def _execute_line_clear(self):
        rows_to_clear = self.line_clear_animation['rows']
        rows_to_clear.sort(reverse=True)
        
        for r in rows_to_clear:
            self.grid[1:r+1, :] = self.grid[0:r, :]
            self.grid[0, :] = 0

    def _get_stack_height(self):
        if np.all(self.grid == 0):
            return 0
        non_empty_rows = np.where(np.any(self.grid != 0, axis=1))[0]
        return self.PLAYFIELD_HEIGHT - non_empty_rows.min()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw playfield border and background
        pygame.draw.rect(self.screen, self.COLOR_GRID, 
                         (self.GRID_X - 2, self.GRID_Y - 2, 
                          self.PLAYFIELD_WIDTH * self.BLOCK_SIZE + 4, 
                          self.PLAYFIELD_HEIGHT * self.BLOCK_SIZE + 4))
        pygame.draw.rect(self.screen, self.COLOR_BG, 
                         (self.GRID_X, self.GRID_Y, 
                          self.PLAYFIELD_WIDTH * self.BLOCK_SIZE, 
                          self.PLAYFIELD_HEIGHT * self.BLOCK_SIZE))

        # Draw placed blocks
        for r in range(self.PLAYFIELD_HEIGHT):
            for c in range(self.PLAYFIELD_WIDTH):
                color_idx = self.grid[r, c]
                if color_idx != 0:
                    self._draw_block(c, r, self.BLOCK_COLORS[color_idx])

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_piece = self.current_piece.copy()
            while self._is_valid_position(ghost_piece, dy=1):
                ghost_piece['y'] += 1
            
            coords = self._get_piece_coords(ghost_piece)
            for r, c in coords:
                if r >= 0:
                    rect = (self.GRID_X + c * self.BLOCK_SIZE, self.GRID_Y + r * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.gfxdraw.box(self.screen, rect, self.COLOR_GHOST)

        # Draw current piece
        if self.current_piece and not self.game_over:
            color_idx = self._rotated_tetrominoes[self.current_piece['name']]['color_idx']
            coords = self._get_piece_coords(self.current_piece)
            for r, c in coords:
                self._draw_block(c, r, self.BLOCK_COLORS[color_idx])
                
        # Draw line clear animation
        if self.line_clear_animation:
            flash_color = (255, 255, 255) if (self.line_clear_animation['timer'] // 2) % 2 == 0 else self.COLOR_BG
            for r in self.line_clear_animation['rows']:
                pygame.draw.rect(self.screen, flash_color,
                                 (self.GRID_X, self.GRID_Y + r * self.BLOCK_SIZE,
                                  self.PLAYFIELD_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE))

    def _draw_block(self, c, r, color):
        if r < 0: return # Don't draw blocks above the playfield
        rect = pygame.Rect(self.GRID_X + c * self.BLOCK_SIZE,
                           self.GRID_Y + r * self.BLOCK_SIZE,
                           self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        # Main color
        pygame.draw.rect(self.screen, color, rect)
        
        # 3D effect with borders
        light_color = tuple(min(255, x + 50) for x in color)
        dark_color = tuple(max(0, x - 50) for x in color)
        pygame.draw.line(self.screen, light_color, rect.topleft, rect.topright, 1)
        pygame.draw.line(self.screen, light_color, rect.topleft, rect.bottomleft, 1)
        pygame.draw.line(self.screen, dark_color, rect.bottomright, rect.topright, 1)
        pygame.draw.line(self.screen, dark_color, rect.bottomright, rect.bottomleft, 1)


    def _render_ui(self):
        # Score and Lines
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        lines_text = self.font_main.render(f"LINES: {self.lines_cleared}/{self.LINES_TO_WIN}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (20, 20))
        self.screen.blit(lines_text, (20, 50))

        # Next Piece Preview
        next_text = self.font_main.render("NEXT", True, self.COLOR_UI_TEXT)
        self.screen.blit(next_text, (self.SCREEN_WIDTH - 150, 20))
        
        shape_data = self._rotated_tetrominoes[self.next_piece_name]
        color_idx = shape_data['color_idx']
        shape = shape_data['shapes'][0]
        
        min_r = min(r for r, c in shape)
        max_r = max(r for r, c in shape)
        min_c = min(c for r, c in shape)
        max_c = max(c for r, c in shape)

        preview_block_size = self.BLOCK_SIZE * 0.8
        offset_x = self.SCREEN_WIDTH - 150 + (4 - (max_c - min_c + 1)) * preview_block_size / 2
        offset_y = 60 + (4 - (max_r - min_r + 1)) * preview_block_size / 2

        for r, c in shape:
            rect = pygame.Rect(offset_x + (c - min_c) * preview_block_size,
                               offset_y + (r - min_r) * preview_block_size,
                               preview_block_size, preview_block_size)
            pygame.draw.rect(self.screen, self.BLOCK_COLORS[color_idx], rect)
            pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Game Over Text
        if self.game_over:
            self._render_centered_text("GAME OVER", 60, 0)
        elif self.lines_cleared >= self.LINES_TO_WIN:
            self._render_centered_text("YOU WIN!", 60, 0)

    def _render_centered_text(self, text, size, y_offset):
        font = pygame.font.SysFont("Consolas", size, bold=True)
        text_surface = font.render(text, True, self.COLOR_UI_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + y_offset))
        
        # Add a dark background for readability
        bg_rect = text_rect.inflate(20, 10)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 180))
        self.screen.blit(bg_surf, bg_rect)
        
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared,
            "game_over": self.game_over
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


# Example of how to run the environment
if __name__ == "__main__":
    # For human play
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Set up a window to display the game
    pygame.display.set_caption("Gymnasium Tetris")
    display_screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    running = True
    total_reward = 0
    
    # Map Pygame keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP:    [1, 0, 0],
        pygame.K_DOWN:  [2, 0, 0],
        pygame.K_LEFT:  [3, 0, 0],
        pygame.K_RIGHT: [4, 0, 0],
    }
    
    while running:
        movement_action = 0
        space_action = 0
        shift_action = 0

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        
        # Prioritize left/right over up/down for movement action
        if keys[pygame.K_LEFT]:
            movement_action = 3
        elif keys[pygame.K_RIGHT]:
            movement_action = 4
        elif keys[pygame.K_UP]:
            movement_action = 1
        elif keys[pygame.K_DOWN]:
            movement_action = 2

        if keys[pygame.K_SPACE]:
            space_action = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_action = 1
            
        action = [movement_action, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Episode finished. Total reward: {total_reward}")
            print(f"Info: {info}")
            total_reward = 0
            obs, info = env.reset()
            # Add a small delay before restarting
            pygame.time.wait(2000)

        # Control the frame rate
        env.clock.tick(30)
        
    env.close()