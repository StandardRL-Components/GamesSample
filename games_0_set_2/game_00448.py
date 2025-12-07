
# Generated: 2025-08-27T13:41:12.025810
# Source Brief: brief_00448.md
# Brief Index: 448

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import os
import pygame


# Set SDL to dummy to run headless
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Hold space for soft drop, press shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated falling block puzzle. Clear lines to score, but watch out for the rising stack!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2 - 80
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2

        self.MAX_STEPS = 1000
        self.WIN_CONDITION_LINES = 10

        # Colors
        self.COLOR_BG = (25, 25, 35)
        self.COLOR_GRID = (40, 40, 55)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_FLASH = (255, 255, 255)
        self.PIECE_COLORS = [
            (0, 255, 255),  # I piece (Cyan)
            (255, 255, 0),  # O piece (Yellow)
            (128, 0, 128),  # T piece (Purple)
            (0, 255, 0),    # S piece (Green)
        ]

        # Define piece shapes and rotations
        self._define_pieces()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Initialize state variables
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.piece_bag = []
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0
        self.fall_timer = 0
        self.line_clear_animation = None
        self.last_reward = 0
        self.is_hard_dropping = False
        
        self.reset()
        
        # self.validate_implementation() # Optional validation call

    def _define_pieces(self):
        # 0: I, 1: O, 2: T, 3: S
        self.PIECES = [
            # I piece
            [[[0,1],[1,1],[2,1],[3,1]], [[1,0],[1,1],[1,2],[1,3]]],
            # O piece
            [[[0,0],[0,1],[1,0],[1,1]]],
            # T piece
            [[[0,1],[1,1],[2,1],[1,0]], [[1,0],[1,1],[1,2],[0,1]], [[0,1],[1,1],[2,1],[1,2]], [[1,0],[1,1],[1,2],[2,1]]],
            # S piece
            [[[1,0],[2,0],[0,1],[1,1]], [[0,0],[0,1],[1,1],[1,2]]]
        ]

    def _new_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.PIECES)))
            self.np_random.shuffle(self.piece_bag)
        
        piece_type = self.piece_bag.pop()
        return {
            "type": piece_type,
            "rotation": 0,
            "x": self.GRID_WIDTH // 2 - 2,
            "y": 0,
            "color": self.PIECE_COLORS[piece_type]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        self.piece_bag = []
        self.current_piece = self._new_piece()
        self.next_piece = self._new_piece()
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_speed = 0.5  # seconds per grid cell
        self.fall_timer = 0
        self.line_clear_animation = None
        self.last_reward = 0
        self.is_hard_dropping = False

        if self._check_collision(self.current_piece, 0, 0):
            self.game_over = True

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        self.steps += 1
        self.last_reward = -0.02 # Per-step penalty

        if self.game_over:
            return self._get_observation(), self.last_reward, True, False, self._get_info()

        # Handle line clear animation delay
        if self.line_clear_animation and self.line_clear_animation['timer'] > 0:
            self.line_clear_animation['timer'] -= self.clock.get_time() / 1000.0
            if self.line_clear_animation['timer'] <= 0:
                self._execute_line_clear()
            # No other actions during animation
            return self._get_observation(), self.last_reward, False, False, self._get_info()

        # --- Action Handling ---
        # Handle hard drop first as it's a one-shot action
        if shift_held and not self.is_hard_dropping:
            self.is_hard_dropping = True
            # Find landing spot and move piece
            drop_y = 0
            while not self._check_collision(self.current_piece, 0, drop_y + 1):
                drop_y += 1
            self.current_piece['y'] += drop_y
            self._lock_piece()
            # SFX: Hard drop thud
        else:
            if not shift_held:
                self.is_hard_dropping = False
            
            # Handle movement and rotation
            self._handle_input(movement)
        
            # --- Game Logic Update ---
            self.fall_timer += self.clock.get_time() / 1000.0
            
            # Soft drop increases fall speed
            current_fall_speed = self.fall_speed / 5.0 if space_held else self.fall_speed

            if self.fall_timer >= current_fall_speed:
                self.fall_timer = 0
                self._move_piece(0, 1)

        terminated = self._check_termination()
        if terminated and self.game_over:
             self.last_reward -= 100
        elif terminated and self.lines_cleared >= self.WIN_CONDITION_LINES:
             self.last_reward += 100

        self.clock.tick(30) # Maintain 30 FPS for smooth auto-advance

        return self._get_observation(), self.last_reward, terminated, False, self._get_info()

    def _handle_input(self, movement):
        # 1=Rotate CW, 2=Rotate CCW, 3=Left, 4=Right
        if movement == 1: # Rotate Clockwise
            self._rotate_piece(1)
        elif movement == 2: # Rotate Counter-Clockwise
            self._rotate_piece(-1)
        elif movement == 3: # Move Left
            self._move_piece(-1, 0)
        elif movement == 4: # Move Right
            self._move_piece(1, 0)

    def _rotate_piece(self, direction):
        if self.current_piece is None: return
        
        piece = self.current_piece
        original_rotation = piece['rotation']
        
        num_rotations = len(self.PIECES[piece['type']])
        piece['rotation'] = (piece['rotation'] + direction) % num_rotations
        
        if self._check_collision(piece, 0, 0):
            # Wall kick attempt
            if not self._check_collision(piece, 1, 0):
                piece['x'] += 1
                # SFX: Rotate success
            elif not self._check_collision(piece, -1, 0):
                piece['x'] -= 1
                # SFX: Rotate success
            else: # Failed
                piece['rotation'] = original_rotation
                # SFX: Rotate fail
        else:
            # SFX: Rotate success
            pass

    def _move_piece(self, dx, dy):
        if self.current_piece is None: return
        
        if not self._check_collision(self.current_piece, dx, dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
        elif dy > 0: # Collision while moving down
            self._lock_piece()
            # SFX: Block lock

    def _check_collision(self, piece, dx, dy):
        shape = self.PIECES[piece['type']][piece['rotation']]
        for x, y in shape:
            new_x = piece['x'] + x + dx
            new_y = piece['y'] + y + dy
            if (new_x < 0 or new_x >= self.GRID_WIDTH or
                new_y < 0 or new_y >= self.GRID_HEIGHT or
                self.grid[new_y, new_x] > 0):
                return True
        return False

    def _lock_piece(self):
        if self.current_piece is None: return

        shape = self.PIECES[self.current_piece['type']][self.current_piece['rotation']]
        
        # Check for risky placement
        highest_point = self.GRID_HEIGHT
        for _, y in shape:
            highest_point = min(highest_point, self.current_piece['y'] + y)
        if highest_point <= 3:
            self.last_reward += 0.1
            self.score += 10 # Bonus score for risky play

        # Place piece on grid
        for x, y in shape:
            grid_x, grid_y = self.current_piece['x'] + x, self.current_piece['y'] + y
            if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                self.grid[grid_y, grid_x] = self.current_piece['type'] + 1

        self._check_line_clears()

        # Spawn new piece
        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()

        # Check for game over
        if self._check_collision(self.current_piece, 0, 0):
            self.game_over = True
            self.current_piece = None # Don't draw the piece that caused the loss

    def _check_line_clears(self):
        lines_to_clear = []
        for y in range(self.GRID_HEIGHT):
            if np.all(self.grid[y, :] > 0):
                lines_to_clear.append(y)

        if lines_to_clear:
            self.line_clear_animation = {'lines': lines_to_clear, 'timer': 0.15}
            # SFX: Line clear trigger
        
    def _execute_line_clear(self):
        lines = self.line_clear_animation['lines']
        num_cleared = len(lines)
        
        # Reward and score
        self.last_reward += num_cleared * 1.0
        self.score += (100 * num_cleared) * num_cleared # Bonus for multi-line clears

        # Shift grid down
        for line_y in sorted(lines, reverse=False):
            self.grid[1:line_y+1, :] = self.grid[0:line_y, :]
            self.grid[0, :] = 0
        
        self.lines_cleared += num_cleared

        # Increase difficulty
        self.fall_speed = max(0.1, 0.5 - 0.05 * (self.lines_cleared // 2))

        self.line_clear_animation = None
        # SFX: Lines disappear

    def _check_termination(self):
        return self.game_over or self.lines_cleared >= self.WIN_CONDITION_LINES or self.steps >= self.MAX_STEPS

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines": self.lines_cleared}

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid background
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (self.GRID_X + x * self.CELL_SIZE, self.GRID_Y + y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

        # Draw placed blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] > 0:
                    self._draw_block(x, y, self.PIECE_COLORS[self.grid[y, x] - 1])

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            ghost_piece = self.current_piece.copy()
            drop_y = 0
            while not self._check_collision(ghost_piece, 0, drop_y + 1):
                drop_y += 1
            ghost_piece['y'] += drop_y
            shape = self.PIECES[ghost_piece['type']][ghost_piece['rotation']]
            for x, y in shape:
                self._draw_block(ghost_piece['x'] + x, ghost_piece['y'] + y, ghost_piece['color'], is_ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            shape = self.PIECES[self.current_piece['type']][self.current_piece['rotation']]
            for x, y in shape:
                self._draw_block(self.current_piece['x'] + x, self.current_piece['y'] + y, self.current_piece['color'])
        
        # Draw line clear animation
        if self.line_clear_animation and self.line_clear_animation['timer'] > 0:
            for line_y in self.line_clear_animation['lines']:
                rect = (self.GRID_X, self.GRID_Y + line_y * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_FLASH, rect)

    def _draw_block(self, grid_x, grid_y, color, is_ghost=False):
        screen_x = self.GRID_X + grid_x * self.CELL_SIZE
        screen_y = self.GRID_Y + grid_y * self.CELL_SIZE
        rect = pygame.Rect(screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)

        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2) # Just the outline
        else:
            # Main block color
            pygame.draw.rect(self.screen, color, rect)
            # Border for definition
            pygame.draw.rect(self.screen, tuple(c*0.7 for c in color), rect, 1)


    def _render_ui(self):
        # Score display
        score_text = self.font_medium.render(f"SCORE", True, self.COLOR_TEXT)
        score_val = self.font_large.render(f"{self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (30, 30))
        self.screen.blit(score_val, (30, 60))

        # Lines display
        lines_text = self.font_medium.render(f"LINES", True, self.COLOR_TEXT)
        lines_val = self.font_large.render(f"{self.lines_cleared} / {self.WIN_CONDITION_LINES}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (30, 130))
        self.screen.blit(lines_val, (30, 160))

        # Next piece preview
        next_text = self.font_medium.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (self.WIDTH - 150, 30))
        if self.next_piece:
            shape = self.PIECES[self.next_piece['type']][0] # Always show default rotation
            for x, y in shape:
                screen_x = self.WIDTH - 140 + x * self.CELL_SIZE
                screen_y = 80 + y * self.CELL_SIZE
                rect = pygame.Rect(screen_x, screen_y, self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.next_piece['color'], rect)
                pygame.draw.rect(self.screen, tuple(c*0.7 for c in self.next_piece['color']), rect, 1)

        # Game Over / Win message
        if self.game_over:
            self._render_overlay_message("GAME OVER")
        elif self.lines_cleared >= self.WIN_CONDITION_LINES:
            self._render_overlay_message("YOU WIN!")

    def _render_overlay_message(self, message):
        overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        text = self.font_large.render(message, True, self.COLOR_FLASH)
        text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(text, text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    # It will not be executed by the Gymnasium environment runner
    
    # Re-enable video driver for direct play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Falling Block Puzzle")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        
        # One-shot key presses for rotation
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    movement = 1 # Rotate CW
                elif event.key == pygame.K_DOWN:
                    movement = 2 # Rotate CCW
                elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = 1 # Hard drop

        # Continuous key presses for movement
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Match env's internal clock

    pygame.quit()