
# Generated: 2025-08-28T01:44:16.297274
# Source Brief: brief_04210.md
# Brief Index: 4210

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←/→ to move, ↑/↓ to rotate. Hold Space for soft drop, Shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced puzzle game. Complete horizontal lines with falling blocks to score points before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Screen and grid dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 10
        self.CELL_SIZE = 30
        
        # Centering the grid with space for UI on the right
        self.GRID_X_OFFSET = (self.SCREEN_WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2 - 80
        self.GRID_Y_OFFSET = (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

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
        self.font_main = pygame.font.Font(None, 36)
        self.font_title = pygame.font.Font(None, 48)

        # Colors and visual styles
        self._define_colors()
        self._define_pieces()
        
        # State variables are initialized in reset()
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.fall_counter = 0.0
        self.base_fall_speed = 0.05
        self.current_fall_speed = 0.0
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        self.last_action_time = 0 # for input cooldown
        self.input_cooldown = 3 # frames

        # Initialize state
        self.reset()
        
        # Run validation
        self.validate_implementation()

    def _define_colors(self):
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_DANGER = (120, 20, 20, 100)
        self.COLOR_GHOST = (255, 255, 255, 60)
        self.COLOR_WHITE = (240, 240, 240)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_SCORE = (255, 215, 0)
        self.PIECE_COLORS = [
            (0, 0, 0),  # 0 is empty
            (230, 60, 60),    # T-piece (Red)
            (60, 230, 60),    # L-piece (Green)
            (60, 60, 230),    # I-piece (Blue)
            (230, 230, 60),   # O-piece (Yellow)
        ]

    def _define_pieces(self):
        # Piece shapes defined by coordinates relative to a pivot
        self.PIECE_SHAPES = {
            1: [[(0, 0), (-1, 0), (1, 0), (0, -1)]],  # T
            2: [[(0, 0), (-1, 0), (1, 0), (1, -1)]],  # L
            3: [[(0, 0), (-1, 0), (1, 0), (2, 0)]],   # I
            4: [[(0, 0), (1, 0), (0, 1), (1, 1)]],    # O
        }
        # Pre-calculate all rotations
        for shape_id in list(self.PIECE_SHAPES.keys()):
            base_shape = self.PIECE_SHAPES[shape_id][0]
            rotations = [base_shape]
            current_shape = base_shape
            for _ in range(3):
                # Rotate 90 degrees clockwise: (x, y) -> (-y, x)
                rotated_shape = [(-y, x) for x, y in current_shape]
                # Sort to maintain a canonical representation
                rotated_shape.sort()
                # Only add if it's a new rotation (O-piece has 1)
                if rotated_shape not in rotations:
                    rotations.append(rotated_shape)
                current_shape = rotated_shape
            self.PIECE_SHAPES[shape_id] = rotations

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.fall_counter = 0.0
        self.current_fall_speed = self.base_fall_speed
        self.lines_to_clear = []
        self.clear_animation_timer = 0
        
        self._spawn_new_piece() # For next_piece
        self._spawn_new_piece() # For current_piece
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = self.game_over

        if not terminated:
            # Handle line clear animation state
            if self.clear_animation_timer > 0:
                self.clear_animation_timer -= 1
                if self.clear_animation_timer == 0:
                    lines_cleared = self._execute_line_clear()
                    reward += lines_cleared * 1.0 # +1 per line cleared
            else:
                # Process player input and game physics
                reward += self._handle_input(action)
                fall_reward = self._update_fall(action)
                reward += fall_reward

        # Update game state and check for termination
        self.steps += 1
        if self.score >= 1000 and not terminated:
            terminated = True
            reward += 100  # Goal-oriented reward for winning
        if self.game_over and not terminated:
            terminated = True
            reward -= 50 # Penalty for losing
        
        if self.steps >= 10000:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        if self.game_over: return 0

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # Cooldown to prevent overly sensitive rotations/movements
        if self.steps < self.last_action_time + self.input_cooldown:
             # Hard drop and soft drop can bypass the cooldown
            if movement != 0:
                movement = 0
        
        if shift_held:
            # Hard drop
            # Find landing position
            final_y = self.current_piece['y']
            while not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], final_y + 1)):
                final_y += 1
            self.current_piece['y'] = final_y
            # Sound effect placeholder: # sfx_hard_drop
            return self._lock_piece()

        if movement != 0:
            self.last_action_time = self.steps

        if movement == 1: # Rotate CW
            self._rotate_piece(1)
        elif movement == 2: # Rotate CCW
            self._rotate_piece(-1)
        elif movement == 3: # Move Left
            self._move_piece(-1)
        elif movement == 4: # Move Right
            self._move_piece(1)
        
        return 0

    def _rotate_piece(self, direction):
        piece = self.current_piece
        current_rotation = piece['rotation']
        num_rotations = len(self.PIECE_SHAPES[piece['id']])
        new_rotation = (current_rotation + direction) % num_rotations
        
        new_shape = self.PIECE_SHAPES[piece['id']][new_rotation]
        
        if not self._check_collision(new_shape, (piece['x'], piece['y'])):
            piece['rotation'] = new_rotation
            piece['shape'] = new_shape
            # Sound effect placeholder: # sfx_rotate

    def _move_piece(self, dx):
        piece = self.current_piece
        if not self._check_collision(piece['shape'], (piece['x'] + dx, piece['y'])):
            piece['x'] += dx
            # Sound effect placeholder: # sfx_move

    def _update_fall(self, action):
        space_held = action[1] == 1
        
        # Difficulty scaling
        self.current_fall_speed = self.base_fall_speed + (self.score // 200) * 0.05
        
        # Soft drop
        fall_increment = self.current_fall_speed * 5 if space_held else self.current_fall_speed
        
        self.fall_counter += fall_increment
        if self.fall_counter >= 1.0:
            self.fall_counter = 0.0
            piece = self.current_piece
            if not self._check_collision(piece['shape'], (piece['x'], piece['y'] + 1)):
                piece['y'] += 1
            else:
                return self._lock_piece()
        return 0

    def _lock_piece(self):
        reward = 0.1  # Base reward for placing a block
        piece = self.current_piece
        
        empty_cells_below = 0
        risky_placements = 0

        for x_off, y_off in piece['shape']:
            grid_x, grid_y = piece['x'] + x_off, piece['y'] + y_off
            if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT:
                self.grid[grid_y, grid_x] = piece['id']
                
                # Check for empty cells directly below
                if grid_y + 1 < self.GRID_HEIGHT and self.grid[grid_y + 1, grid_x] == 0:
                    empty_cells_below += 1
                
                # Check if this placement created a new hole (risky placement)
                # A hole is an empty cell with a filled one above it.
                # Here, we check if we just placed a block over an empty cell.
                if grid_y > 0 and self.grid[grid_y - 1, grid_x] == 0:
                    # This check is flawed. Let's re-evaluate "leaving single empty cells below"
                    # A better interpretation: check for empty cells directly below the placed piece.
                    # This is already covered by `empty_cells_below`. Let's use that for "risk".
                    pass

        # Reward calculation based on brief
        reward -= 0.02 * empty_cells_below
        # Let's interpret "risky placements" as those that create many empty cells below
        if empty_cells_below > 2:
            reward += 5.0

        # Check for line clears
        lines_found = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y, :] != 0):
                lines_found.append(y)
        
        if lines_found:
            self.lines_to_clear = lines_found
            self.clear_animation_timer = 5 # frames for animation
            # Sound effect placeholder: # sfx_line_pending
        else:
            self._spawn_new_piece()

        return reward

    def _execute_line_clear(self):
        num_cleared = len(self.lines_to_clear)
        if num_cleared == 0: return 0

        # Remove lines from bottom up
        for y in sorted(self.lines_to_clear, reverse=True):
            self.grid = np.delete(self.grid, y, axis=0)
        
        # Add new empty lines at the top
        new_lines = np.zeros((num_cleared, self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack([new_lines, self.grid])
        
        # Score update
        score_map = {1: 10, 2: 30, 3: 60, 4: 120}
        self.score += score_map.get(num_cleared, 0) * 10 # scale up to reach 1000
        
        self.lines_to_clear = []
        self._spawn_new_piece()
        # Sound effect placeholder: # sfx_line_clear
        return num_cleared

    def _spawn_new_piece(self):
        if self.next_piece:
            self.current_piece = self.next_piece
        
        new_id = self.np_random.integers(1, len(self.PIECE_SHAPES) + 1)
        new_shape_rotations = self.PIECE_SHAPES[new_id]
        
        self.next_piece = {
            'id': new_id,
            'rotation': 0,
            'shape': new_shape_rotations[0],
            'x': self.GRID_WIDTH // 2,
            'y': 0, # Start at the very top
            'color': self.PIECE_COLORS[new_id]
        }
        
        # After spawning, check for game over
        if self.current_piece and self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
            self.game_over = True
            # Sound effect placeholder: # sfx_game_over

    def _check_collision(self, shape, pos):
        px, py = pos
        for x_off, y_off in shape:
            grid_x, grid_y = px + x_off, py + y_off
            # Check boundaries
            if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                return True
            # Check against locked blocks
            if self.grid[grid_y, grid_x] != 0:
                return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        self._render_grid_background()
        self._render_danger_zone()
        self._render_placed_blocks()
        
        if not self.game_over:
            self._render_ghost_piece()
            self._render_piece(self.current_piece, self.GRID_X_OFFSET, self.GRID_Y_OFFSET)
        
        if self.clear_animation_timer > 0:
            self._render_line_clear_animation()

    def _render_grid_background(self):
        pygame.draw.rect(
            self.screen, self.COLOR_GRID,
            (self.GRID_X_OFFSET, self.GRID_Y_OFFSET, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE)
        )
        for x in range(self.GRID_WIDTH + 1):
            start_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET)
            end_pos = (self.GRID_X_OFFSET + x * self.CELL_SIZE, self.GRID_Y_OFFSET + self.GRID_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_BG, start_pos, end_pos)
        for y in range(self.GRID_HEIGHT + 1):
            start_pos = (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            end_pos = (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE, self.GRID_Y_OFFSET + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_BG, start_pos, end_pos)

    def _render_danger_zone(self):
        s = pygame.Surface((self.GRID_WIDTH * self.CELL_SIZE, 2 * self.CELL_SIZE), pygame.SRCALPHA)
        s.fill(self.COLOR_DANGER)
        self.screen.blit(s, (self.GRID_X_OFFSET, self.GRID_Y_OFFSET))

    def _render_placed_blocks(self):
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell_id = self.grid[y, x]
                if cell_id != 0:
                    self._draw_cell(x, y, self.PIECE_COLORS[cell_id], self.GRID_X_OFFSET, self.GRID_Y_OFFSET)

    def _render_piece(self, piece, x_base, y_base):
        if not piece: return
        for x_off, y_off in piece['shape']:
            self._draw_cell(piece['x'] + x_off, piece['y'] + y_off, piece['color'], x_base, y_base)

    def _render_ghost_piece(self):
        if not self.current_piece: return
        ghost_piece = self.current_piece.copy()
        
        # Find landing position
        final_y = ghost_piece['y']
        while not self._check_collision(ghost_piece['shape'], (ghost_piece['x'], final_y + 1)):
            final_y += 1
        
        # Draw ghost
        s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
        for x_off, y_off in ghost_piece['shape']:
            s.fill(self.COLOR_GHOST)
            px = self.GRID_X_OFFSET + (ghost_piece['x'] + x_off) * self.CELL_SIZE
            py = self.GRID_Y_OFFSET + (final_y + y_off) * self.CELL_SIZE
            self.screen.blit(s, (px, py))

    def _render_line_clear_animation(self):
        # Flash effect
        if self.clear_animation_timer % 2 == 0: return
        for y in self.lines_to_clear:
            pygame.draw.rect(
                self.screen, self.COLOR_WHITE,
                (self.GRID_X_OFFSET, self.GRID_Y_OFFSET + y * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
            )

    def _render_ui(self):
        # Score display
        score_text = self.font_title.render(f"{self.score}", True, self.COLOR_SCORE)
        score_label = self.font_main.render("SCORE", True, self.COLOR_TEXT)
        self.screen.blit(score_label, (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y_OFFSET + 120))
        self.screen.blit(score_text, (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y_OFFSET + 150))

        # Next piece preview
        next_label = self.font_main.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_label, (self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE + 20, self.GRID_Y_OFFSET + 20))
        preview_x = self.GRID_X_OFFSET + self.GRID_WIDTH * self.CELL_SIZE + 30
        preview_y = self.GRID_Y_OFFSET + 60
        if self.next_piece:
            # Center the piece in the preview area
            centered_piece = self.next_piece.copy()
            centered_piece['x'] = 1
            centered_piece['y'] = 1
            self._render_piece(centered_piece, preview_x, preview_y)

        # Game Over text
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            status_text = "YOU WIN!" if self.score >= 1000 else "GAME OVER"
            game_over_surf = self.font_title.render(status_text, True, self.COLOR_WHITE)
            text_rect = game_over_surf.get_rect(center=(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2))
            self.screen.blit(game_over_surf, text_rect)

    def _draw_cell(self, x, y, color, x_base, y_base):
        px = x_base + x * self.CELL_SIZE
        py = y_base + y * self.CELL_SIZE
        
        # Beveled effect
        outer_color = tuple(max(0, c - 50) for c in color)
        pygame.draw.rect(self.screen, outer_color, (px, py, self.CELL_SIZE, self.CELL_SIZE))
        pygame.draw.rect(self.screen, color, (px + 2, py + 2, self.CELL_SIZE - 4, self.CELL_SIZE - 4))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.score // 100, # Approximation
        }

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
if __name__ == '__main__':
    env = GameEnv(render_mode="rgb_array")
    
    # To play manually
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy" # Ensure headless for server
    
    try:
        os.environ["SDL_VIDEODRIVER"] = "x11" # Try to show a window
        screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
        pygame.display.set_caption("Puzzle Game")
    except pygame.error:
        print("Display not found, running headlessly.")
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        screen = None

    obs, info = env.reset()
    terminated = False
    clock = pygame.time.Clock()
    
    # Action state
    movement = 0
    space_held = 0
    shift_held = 0

    print(env.user_guide)

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
            
            # Key down events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    movement = 3
                elif event.key == pygame.K_RIGHT:
                    movement = 4
                elif event.key == pygame.K_UP:
                    movement = 1
                elif event.key == pygame.K_DOWN:
                    movement = 2
                elif event.key == pygame.K_SPACE:
                    space_held = 1
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1
            
            # Key up events
            if event.type == pygame.KEYUP:
                if event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                    movement = 0
                if event.key == pygame.K_SPACE:
                    space_held = 0
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 0
        
        # Construct action and step
        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if screen:
            # Draw the observation to the display window
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()

        # Reset movement for next frame if it's a one-shot action
        if movement in [1, 2, 3, 4]:
            movement = 0
        # Hard drop is a single event
        if shift_held == 1:
            shift_held = 0

        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}")
    pygame.quit()