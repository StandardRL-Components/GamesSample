
# Generated: 2025-08-27T19:31:32.678409
# Source Brief: brief_02178.md
# Brief Index: 2178

        
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
        "Controls: ←→ to move, ↑ to rotate clockwise, ↓ to soft drop. "
        "Hold Shift to rotate counter-clockwise. Press Space for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced, procedurally generated falling block puzzle. "
        "Clear 10 lines to win. Place blocks with overhangs for bonus points."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.PLAYFIELD_WIDTH = 10
        self.PLAYFIELD_HEIGHT = 20
        self.CELL_SIZE = 18
        self.MAX_STEPS = 1000
        self.LINES_TO_WIN = 10
        
        # Centering the playfield
        self.GRID_TOP_LEFT_X = (self.SCREEN_WIDTH - self.PLAYFIELD_WIDTH * self.CELL_SIZE) // 2
        self.GRID_TOP_LEFT_Y = (self.SCREEN_HEIGHT - self.PLAYFIELD_HEIGHT * self.CELL_SIZE) // 2

        # Timing
        self.GRAVITY_RATE = 15  # Ticks per fall
        self.SOFT_DROP_RATE = 2 # Ticks per fall when soft dropping

        # --- Colors ---
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_GHOST = (255, 255, 255, 50) # RGBA for transparency
        self.COLOR_FLASH = (255, 255, 255)
        
        # Tetromino shapes and colors
        self.PIECE_SHAPES = {
            'T': [[[1, 1, 1], [0, 1, 0]]],
            'I': [[[1, 1, 1, 1]]],
            'O': [[[1, 1], [1, 1]]],
            'L': [[[1, 0, 0], [1, 1, 1]]],
            'J': [[[0, 0, 1], [1, 1, 1]]],
            'S': [[[0, 1, 1], [1, 1, 0]]],
            'Z': [[[1, 1, 0], [0, 1, 1]]]
        }
        self.PIECE_COLORS = {
            'T': (160, 0, 255),  # Purple
            'I': (0, 255, 255),  # Cyan
            'O': (255, 255, 0),  # Yellow
            'L': (255, 165, 0),  # Orange
            'J': (0, 0, 255),    # Blue
            'S': (0, 255, 0),    # Green
            'Z': (255, 0, 0)     # Red
        }
        # Generate all rotations for each piece
        for shape_name in self.PIECE_SHAPES:
            base_shape = np.array(self.PIECE_SHAPES[shape_name][0])
            rotations = [base_shape]
            for _ in range(3):
                base_shape = np.rot90(base_shape)
                rotations.append(base_shape)
            self.PIECE_SHAPES[shape_name] = rotations

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 18)
        
        # State variables (initialized in reset)
        self.playfield = None
        self.current_piece = None
        self.next_piece = None
        self.piece_bag = []
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.gravity_timer = 0
        self.flash_timer = 0
        self.flashing_lines = []

        # Action handling state
        self.prev_space_held = False
        self.prev_shift_held = False
        self.prev_up_pressed = False
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.playfield = np.zeros((self.PLAYFIELD_HEIGHT, self.PLAYFIELD_WIDTH), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.gravity_timer = 0
        self.flash_timer = 0
        self.flashing_lines = []

        self.prev_space_held = True # Prevent action on first frame
        self.prev_shift_held = True
        self.prev_up_pressed = True

        self._fill_piece_bag()
        self._spawn_piece() # Spawns current
        self._spawn_piece() # Spawns next

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = -0.1  # Small penalty per step to encourage efficiency
        piece_locked_this_step = False

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle one-shot actions (to prevent repeated actions on hold) ---
        is_hard_drop = space_held and not self.prev_space_held
        is_rotate_cw = (movement == 1) and not self.prev_up_pressed
        is_rotate_ccw = shift_held and not self.prev_shift_held
        
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held
        self.prev_up_pressed = (movement == 1)

        # --- Update game logic ---

        # 1. Hard Drop (takes precedence)
        if is_hard_drop:
            # sfx: hard_drop_sound
            self.current_piece['y'] = self._get_ghost_y()
            piece_locked_this_step = True

        # 2. Rotation & Movement (if not hard dropping)
        if not piece_locked_this_step:
            if is_rotate_cw: self._rotate_piece(1)
            if is_rotate_ccw: self._rotate_piece(-1)
            
            if movement == 3: self._move_piece(-1, 0) # Left
            if movement == 4: self._move_piece(1, 0) # Right

        # 3. Gravity
        if not piece_locked_this_step:
            drop_speed = self.SOFT_DROP_RATE if movement == 2 else self.GRAVITY_RATE
            self.gravity_timer += 1
            if self.gravity_timer >= drop_speed:
                self.gravity_timer = 0
                if not self._move_piece(0, 1):
                    piece_locked_this_step = True # Piece hit something below

        # 4. Lock Piece Logic
        if piece_locked_this_step:
            # sfx: piece_lock_sound
            reward += self._calculate_placement_reward()
            self._place_piece()
            
            cleared_count = self._check_lines()
            if cleared_count > 0:
                # sfx: line_clear_sound
                reward += cleared_count # +1 per line
                self.lines_cleared += cleared_count
                self.score += [0, 100, 300, 500, 800][cleared_count] # Standard scoring

            self._spawn_piece()
            if not self._is_valid_position(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y']):
                self.game_over = True

        self.steps += 1
        
        # --- Termination and Final Rewards ---
        terminated = self.game_over or self.lines_cleared >= self.LINES_TO_WIN or self.steps >= self.MAX_STEPS
        if self.game_over:
            # sfx: game_over_sound
            reward += -100
        elif self.lines_cleared >= self.LINES_TO_WIN:
            # sfx: victory_sound
            reward += 100

        return self._get_observation(), reward, terminated, False, self._get_info()

    # --- Helper Functions ---

    def _fill_piece_bag(self):
        self.piece_bag = list(self.PIECE_SHAPES.keys())
        self.np_random.shuffle(self.piece_bag)

    def _spawn_piece(self):
        if not self.piece_bag:
            self._fill_piece_bag()
        
        piece_type = self.piece_bag.pop(0)
        shape = self.PIECE_SHAPES[piece_type][0]
        
        self.current_piece = self.next_piece
        self.next_piece = {
            'type': piece_type,
            'shape': shape,
            'rotation': 0,
            'x': self.PLAYFIELD_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'color': self.PIECE_COLORS[piece_type]
        }

    def _move_piece(self, dx, dy):
        if self._is_valid_position(self.current_piece['shape'], self.current_piece['x'] + dx, self.current_piece['y'] + dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def _rotate_piece(self, direction):
        # sfx: rotate_sound
        piece = self.current_piece
        current_rotation = piece['rotation']
        new_rotation = (current_rotation + direction) % 4
        new_shape = self.PIECE_SHAPES[piece['type']][new_rotation]

        # Simple wall kick logic
        for dx in [0, -1, 1, -2, 2]: # Test current pos, then kick left/right
            if self._is_valid_position(new_shape, piece['x'] + dx, piece['y']):
                piece['shape'] = new_shape
                piece['rotation'] = new_rotation
                piece['x'] += dx
                return

    def _is_valid_position(self, shape, grid_x, grid_y):
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    x, y = grid_x + c, grid_y + r
                    if not (0 <= x < self.PLAYFIELD_WIDTH and 0 <= y < self.PLAYFIELD_HEIGHT):
                        return False # Out of bounds
                    if self.playfield[y, x] != 0:
                        return False # Collision with existing block
        return True

    def _place_piece(self):
        piece = self.current_piece
        shape = piece['shape']
        color_index = list(self.PIECE_COLORS.keys()).index(piece['type']) + 1
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self.playfield[piece['y'] + r, piece['x'] + c] = color_index

    def _check_lines(self):
        lines_to_clear = [r for r, row in enumerate(self.playfield) if np.all(row)]
        if not lines_to_clear:
            return 0
        
        self.flashing_lines = lines_to_clear
        self.flash_timer = 5 # Flash for 5 frames

        # Clear lines and shift down
        cleared_count = len(lines_to_clear)
        self.playfield = np.delete(self.playfield, lines_to_clear, axis=0)
        new_rows = np.zeros((cleared_count, self.PLAYFIELD_WIDTH), dtype=int)
        self.playfield = np.vstack((new_rows, self.playfield))
        return cleared_count

    def _get_ghost_y(self):
        y = self.current_piece['y']
        while self._is_valid_position(self.current_piece['shape'], self.current_piece['x'], y + 1):
            y += 1
        return y

    def _calculate_placement_reward(self):
        piece = self.current_piece
        shape = piece['shape']
        final_y = self._get_ghost_y()
        has_overhang = False

        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    y_pos = final_y + r + 1
                    x_pos = piece['x'] + c
                    if y_pos >= self.PLAYFIELD_HEIGHT: # Piece is on the floor
                        continue
                    if self.playfield[y_pos, x_pos] == 0:
                        has_overhang = True
                        break
            if has_overhang:
                break
        
        return 2.0 if has_overhang else -0.2

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for x in range(self.PLAYFIELD_WIDTH + 1):
            start = (self.GRID_TOP_LEFT_X + x * self.CELL_SIZE, self.GRID_TOP_LEFT_Y)
            end = (self.GRID_TOP_LEFT_X + x * self.CELL_SIZE, self.GRID_TOP_LEFT_Y + self.PLAYFIELD_HEIGHT * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(self.PLAYFIELD_HEIGHT + 1):
            start = (self.GRID_TOP_LEFT_X, self.GRID_TOP_LEFT_Y + y * self.CELL_SIZE)
            end = (self.GRID_TOP_LEFT_X + self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.GRID_TOP_LEFT_Y + y * self.CELL_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

        # Draw placed blocks
        color_keys = list(self.PIECE_COLORS.keys())
        for r in range(self.PLAYFIELD_HEIGHT):
            for c in range(self.PLAYFIELD_WIDTH):
                if self.playfield[r, c] != 0:
                    color = self.PIECE_COLORS[color_keys[int(self.playfield[r, c]) - 1]]
                    self._draw_cell(c, r, color)

        # Draw line clear flash
        if self.flash_timer > 0:
            for r in self.flashing_lines:
                rect = pygame.Rect(self.GRID_TOP_LEFT_X, self.GRID_TOP_LEFT_Y + r * self.CELL_SIZE, self.PLAYFIELD_WIDTH * self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, self.COLOR_FLASH, rect)
            self.flash_timer -= 1
        
        if self.current_piece and not self.game_over:
            # Draw ghost piece
            ghost_y = self._get_ghost_y()
            self._draw_piece(self.current_piece, ghost_y, self.COLOR_GHOST)
            
            # Draw current piece
            self._draw_piece(self.current_piece, self.current_piece['y'], self.current_piece['color'])

    def _draw_piece(self, piece, grid_y, color):
        shape = piece['shape']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_cell(piece['x'] + c, grid_y + r, color)

    def _draw_cell(self, grid_x, grid_y, color):
        px, py = self.GRID_TOP_LEFT_X + grid_x * self.CELL_SIZE, self.GRID_TOP_LEFT_Y + grid_y * self.CELL_SIZE
        rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
        
        # Draw with transparency if specified
        if len(color) == 4:
            s = pygame.Surface((self.CELL_SIZE, self.CELL_SIZE), pygame.SRCALPHA)
            s.fill(color)
            self.screen.blit(s, rect.topleft)
        else:
            pygame.draw.rect(self.screen, color, rect)
        
        # Add a subtle border for definition
        pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)

    def _render_ui(self):
        # Score
        score_text = self.font_main.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (20, 20))

        # Lines
        lines_text = self.font_main.render(f"LINES: {self.lines_cleared} / {self.LINES_TO_WIN}", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (20, 50))

        # Next Piece Preview
        preview_x, preview_y = self.GRID_TOP_LEFT_X + self.PLAYFIELD_WIDTH * self.CELL_SIZE + 30, self.GRID_TOP_LEFT_Y
        next_text = self.font_small.render("NEXT", True, self.COLOR_TEXT)
        self.screen.blit(next_text, (preview_x, preview_y))
        if self.next_piece:
            shape = self.next_piece['shape']
            color = self.next_piece['color']
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        px = preview_x + c * self.CELL_SIZE
                        py = preview_y + 20 + r * self.CELL_SIZE
                        pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE))
                        pygame.draw.rect(self.screen, self.COLOR_GRID, (px, py, self.CELL_SIZE, self.CELL_SIZE), 1)

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            over_text = self.font_main.render("GAME OVER", True, (255, 50, 50))
            text_rect = over_text.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
            self.screen.blit(over_text, text_rect)


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

if __name__ == "__main__":
    # To play the game manually, run this script
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Pygame setup for manual play
    pygame.display.set_caption("Falling Block Puzzle")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    running = True
    while running:
        if terminated:
            # Wait for a key press to reset
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    obs, info = env.reset()
                    terminated = False
        else:
            # --- Action Mapping for Human Play ---
            keys = pygame.key.get_pressed()
            movement = 0 # no-op
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4

            space_held = 1 if keys[pygame.K_SPACE] else 0
            shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
            
            action = [movement, space_held, shift_held]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Frame Rate ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        clock.tick(30) # Run at 30 FPS

    env.close()