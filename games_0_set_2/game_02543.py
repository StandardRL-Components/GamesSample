import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↓ for soft drop. ↑ to rotate clockwise, Shift to rotate counter-clockwise. Space for hard drop."
    )

    game_description = (
        "A fast-paced grid-based puzzle game. Manipulate falling blocks to clear lines and reach the target score before time runs out or the stack reaches the top."
    )

    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 10, 20
    CELL_SIZE = 18
    GRID_WIDTH = GRID_COLS * CELL_SIZE
    GRID_HEIGHT = GRID_ROWS * CELL_SIZE
    GRID_X = (SCREEN_WIDTH - GRID_WIDTH) // 2 - 80
    GRID_Y = (SCREEN_HEIGHT - GRID_HEIGHT) // 2

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_BG = (30, 30, 40)
    COLOR_TIMER_WARN = (255, 100, 100)
    COLOR_GHOST = (255, 255, 255, 50)
    COLOR_WHITE = (255, 255, 255)

    # --- Tetromino Shapes and Colors ---
    SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1, 0], [0, 1, 1]],  # Z
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 1], [0, 1, 0]],  # T
        [[1, 1, 1], [1, 0, 0]],  # L
        [[1, 1, 1], [0, 0, 1]],  # J
        [[1, 1], [1, 1]]   # O
    ]
    COLORS = [
        (0, 240, 240),   # I (Cyan)
        (240, 0, 0),     # Z (Red)
        (0, 240, 0),     # S (Green)
        (160, 0, 240),   # T (Purple)
        (240, 160, 0),   # L (Orange)
        (0, 0, 240),     # J (Blue)
        (240, 240, 0)    # O (Yellow)
    ]

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        try:
            self.font_large = pygame.font.SysFont("ocraextended", 36)
            self.font_medium = pygame.font.SysFont("ocraextended", 24)
            self.font_small = pygame.font.SysFont("ocraextended", 16)
        except pygame.error:
            self.font_large = pygame.font.SysFont("monospace", 30)
            self.font_medium = pygame.font.SysFont("monospace", 20)
            self.font_small = pygame.font.SysFont("monospace", 14)

        self._initialize_state()
        # The validation function is called after initialization, so the state must be valid.
        # We call reset() to ensure the environment is ready for validation and subsequent use.
        self.reset()
        self.validate_implementation()

    def _initialize_state(self):
        # Game state variables are initialized here but fully set in reset()
        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)
        self.piece_bag = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.game_timer = 0

        self.fall_time = 0
        self.fall_counter = 0
        self.initial_fall_speed = 30  # Ticks per grid cell

        self.last_space_held = False
        self.last_shift_held = False
        self.last_up_held = False

        self.line_clear_animation = []
        self.reward_this_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.GRID_ROWS, self.GRID_COLS), dtype=int)

        self.piece_bag = list(range(len(self.SHAPES)))
        self.np_random.shuffle(self.piece_bag)

        self.next_piece = self._new_piece()
        self.current_piece = self._new_piece()

        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.game_timer = 60 * 30  # 60 seconds at 30fps

        self.fall_time = self.initial_fall_speed
        self.fall_counter = 0

        self.last_space_held = False
        self.last_shift_held = False
        self.last_up_held = False

        self.line_clear_animation = []

        return self._get_observation(), self._get_info()

    def _new_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.SHAPES)))
            self.np_random.shuffle(self.piece_bag)

        shape_idx = self.piece_bag.pop()
        shape = self.SHAPES[shape_idx]

        return {
            "shape_idx": shape_idx,
            "shape": shape,
            "rotation": 0,
            "x": self.GRID_COLS // 2 - len(shape[0]) // 2,
            "y": 0,
            "color": self.COLORS[shape_idx]
        }

    def step(self, action):
        self.reward_this_step = -0.01  # Small penalty for time passing

        if not self.game_over:
            self._handle_input(action)
            self._update_game_state()
            self.steps += 1

        reward = self.reward_this_step
        terminated = self._check_termination()

        if terminated and not self.win:
            reward = -100
        elif terminated and self.win:
            reward = 50

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_action, shift_action = action
        space_pressed = space_action == 1 and not self.last_space_held
        shift_pressed = shift_action == 1 and not self.last_shift_held

        # Remap movement[1] (up) to a press event for rotation
        up_pressed = movement == 1 and not self.last_up_held

        # Movement
        if movement == 3:  # Left
            self._move_piece(-1, 0)
        elif movement == 4:  # Right
            self._move_piece(1, 0)

        # Soft drop
        if movement == 2:  # Down
            self.fall_counter += 5  # Accelerate fall
            self.reward_this_step += 0.001  # Tiny reward for faster play

        # Rotation
        if up_pressed:  # Clockwise
            self._rotate_piece(1)
        if shift_pressed:  # Counter-clockwise
            self._rotate_piece(-1)

        # Hard drop
        if space_pressed:
            # sfx: hard_drop_sound
            self._hard_drop()

        self.last_space_held = space_action == 1
        self.last_shift_held = shift_action == 1
        self.last_up_held = movement == 1

    def _update_game_state(self):
        self.game_timer = max(0, self.game_timer - 1)

        if self.line_clear_animation:
            self.line_clear_animation[0]['timer'] -= 1
            if self.line_clear_animation[0]['timer'] <= 0:
                self._clear_lines(self.line_clear_animation.pop(0)['rows'])

        self.fall_counter += 1
        if self.fall_counter >= self.fall_time:
            self.fall_counter = 0
            self._move_piece(0, 1, auto_drop=True)

    def _move_piece(self, dx, dy, auto_drop=False):
        if self.current_piece is None: return

        test_piece = self.current_piece.copy()
        test_piece['x'] += dx
        test_piece['y'] += dy

        if not self._check_collision(test_piece):
            self.current_piece = test_piece
        elif dy > 0 and auto_drop:  # Collision on auto-drop means lock
            self._lock_piece()

    def _rotate_piece(self, direction):
        if self.current_piece is None: return

        # sfx: rotate_sound
        original_shape = self.current_piece['shape']
        rotated_shape = list(zip(*original_shape[::-direction]))
        if direction == -1:  # CCW needs adjustment
            rotated_shape = [list(row) for row in rotated_shape]

        test_piece = self.current_piece.copy()
        test_piece['shape'] = rotated_shape

        # Wall kick logic
        kick_offsets = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1)]
        for ox, oy in kick_offsets:
            test_piece['x'] = self.current_piece['x'] + ox
            test_piece['y'] = self.current_piece['y'] + oy
            if not self._check_collision(test_piece):
                self.current_piece = test_piece
                return

    def _hard_drop(self):
        if self.current_piece is None: return

        dy = 0
        while not self._check_collision(self.current_piece, (0, dy + 1)):
            dy += 1

        self.current_piece['y'] += dy
        self._lock_piece()

    def _lock_piece(self):
        if self.current_piece is None: return

        # sfx: lock_piece_sound
        piece_coords = self._get_piece_coords(self.current_piece)

        # Placement reward calculation
        holes_created = 0
        max_height_before = self._get_max_height()

        for x, y in piece_coords:
            if 0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS:
                self.grid[y, x] = self.current_piece['shape_idx'] + 1
                # Check for holes directly underneath
                if y + 1 < self.GRID_ROWS and self.grid[y + 1, x] == 0:
                    holes_created += 1

        max_height_after = self._get_max_height()
        height_increase = max_height_after - max_height_before

        # Apply placement rewards/penalties
        if holes_created > 0:
            self.reward_this_step += holes_created * 0.5  # Risky placement reward
        if height_increase > 0:
            self.reward_this_step -= height_increase * 0.1  # Penalty for increasing stack height

        lines = self._find_completed_lines()
        if lines:
            # sfx: line_clear_sound
            self.line_clear_animation.append({'rows': lines, 'timer': 5})  # 5 frames animation
            self.lines_cleared += len(lines)

            # Reward for line clears
            self.reward_this_step += len(lines)
            if len(lines) > 1:
                self.reward_this_step += 2  # Bonus for multi-line clear

            self.score += [100, 300, 500, 800][len(lines) - 1]

            # Increase speed
            self.fall_time = max(5, self.initial_fall_speed - (self.lines_cleared // 5) * 0.05 * 30)

        self.current_piece = self.next_piece
        self.next_piece = self._new_piece()

        if self._check_collision(self.current_piece):
            self.game_over = True
            self.current_piece = None  # Stop drawing it

    def _get_max_height(self):
        for r in range(self.GRID_ROWS):
            if np.any(self.grid[r, :]):
                return self.GRID_ROWS - r
        return 0

    def _find_completed_lines(self):
        return [r for r in range(self.GRID_ROWS) if np.all(self.grid[r, :])]

    def _clear_lines(self, lines_to_clear):
        for r in sorted(lines_to_clear, reverse=True):
            self.grid = np.delete(self.grid, r, axis=0)
            self.grid = np.insert(self.grid, 0, np.zeros(self.GRID_COLS), axis=0)

    def _check_collision(self, piece, offset=(0, 0)):
        if piece is None: return True
        coords = self._get_piece_coords(piece, offset)
        for x, y in coords:
            if not (0 <= x < self.GRID_COLS and 0 <= y < self.GRID_ROWS):
                return True  # Wall collision
            if self.grid[y, x] != 0:
                return True  # Other piece collision
        return False

    def _get_piece_coords(self, piece, offset=(0, 0)):
        coords = []
        shape = piece['shape']
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    coords.append((piece['x'] + c + offset[0], piece['y'] + r + offset[1]))
        return coords

    def _check_termination(self):
        if self.game_over:
            return True
        if self.lines_cleared >= 20:
            self.win = True
            self.game_over = True
            return True
        if self.game_timer <= 0:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        if self.game_over:
            self._render_game_over()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines": self.lines_cleared}

    def _render_game(self):
        # Draw grid background
        grid_rect = pygame.Rect(self.GRID_X, self.GRID_Y, self.GRID_WIDTH, self.GRID_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)

        # Draw locked pieces
        if self.grid is not None:
            for r in range(self.GRID_ROWS):
                for c in range(self.GRID_COLS):
                    if self.grid[r, c] != 0:
                        self._draw_block(c, r, self.COLORS[int(self.grid[r, c]) - 1])

        # Draw ghost piece
        if self.current_piece and not self.game_over:
            dy = 0
            while not self._check_collision(self.current_piece, (0, dy + 1)):
                dy += 1
            ghost_piece = self.current_piece.copy()
            ghost_piece['y'] += dy
            coords = self._get_piece_coords(ghost_piece)
            for x, y in coords:
                self._draw_block(x, y, self.COLOR_GHOST, is_ghost=True)

        # Draw current piece
        if self.current_piece and not self.game_over:
            coords = self._get_piece_coords(self.current_piece)
            for x, y in coords:
                self._draw_block(x, y, self.current_piece['color'])

        # Draw line clear animation
        if self.line_clear_animation:
            for anim in self.line_clear_animation:
                for r in anim['rows']:
                    rect = pygame.Rect(self.GRID_X, self.GRID_Y + r * self.CELL_SIZE, self.GRID_WIDTH, self.CELL_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_WHITE, rect)

        # Draw grid border
        pygame.draw.rect(self.screen, self.COLOR_TEXT, grid_rect, 2)

    def _draw_block(self, grid_x, grid_y, color, is_ghost=False):
        x, y = self.GRID_X + grid_x * self.CELL_SIZE, self.GRID_Y + grid_y * self.CELL_SIZE
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)

        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2)
            return

        # Beveled effect
        highlight = tuple(min(255, c + 50) for c in color)
        shadow = tuple(max(0, c - 50) for c in color)

        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.line(self.screen, highlight, (x, y), (x + self.CELL_SIZE - 1, y), 2)
        pygame.draw.line(self.screen, highlight, (x, y), (x, y + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, shadow, (x + self.CELL_SIZE - 1, y), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, shadow, (x, y + self.CELL_SIZE - 1), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)

    def _render_ui(self):
        # Score
        self._draw_ui_box(20, 20, 180, 80, "SCORE")
        score_text = self.font_large.render(f"{self.score:06d}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (35, 55))

        # Lines
        self._draw_ui_box(self.SCREEN_WIDTH - 200, 20, 180, 80, "LINES")
        lines_text = self.font_large.render(f"{self.lines_cleared:02d} / 20", True, self.COLOR_TEXT)
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - 185, 55))

        # Timer
        timer_seconds = self.game_timer / 30
        timer_color = self.COLOR_TIMER_WARN if timer_seconds < 10 else self.COLOR_TEXT
        timer_text = self.font_medium.render(f"TIME: {timer_seconds:.1f}", True, timer_color)
        text_rect = timer_text.get_rect(centerx=self.SCREEN_WIDTH // 2, y=10)
        self.screen.blit(timer_text, text_rect)

        # Next Piece
        self._draw_ui_box(self.SCREEN_WIDTH - 200, 280, 180, 100, "NEXT")
        if self.next_piece:
            shape = self.next_piece['shape']
            w, h = len(shape[0]), len(shape)
            start_x = self.SCREEN_WIDTH - 200 + (180 - w * self.CELL_SIZE) // 2
            start_y = 280 + 30 + (70 - h * self.CELL_SIZE) // 2
            for r, row in enumerate(shape):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block_at_pixel(start_x + c * self.CELL_SIZE, start_y + r * self.CELL_SIZE, self.next_piece['color'])

    def _draw_ui_box(self, x, y, w, h, title):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, rect)
        pygame.draw.rect(self.screen, self.COLOR_TEXT, rect, 2)
        title_text = self.font_small.render(title, True, self.COLOR_TEXT)
        self.screen.blit(title_text, (x + 10, y + 5))

    def _draw_block_at_pixel(self, x, y, color):
        rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)
        highlight = tuple(min(255, c + 50) for c in color)
        shadow = tuple(max(0, c - 50) for c in color)
        pygame.draw.rect(self.screen, color, rect)
        pygame.draw.line(self.screen, highlight, (x, y), (x + self.CELL_SIZE - 1, y), 2)
        pygame.draw.line(self.screen, highlight, (x, y), (x, y + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, shadow, (x + self.CELL_SIZE - 1, y), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)
        pygame.draw.line(self.screen, shadow, (x, y + self.CELL_SIZE - 1), (x + self.CELL_SIZE - 1, y + self.CELL_SIZE - 1), 2)

    def _render_game_over(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        message = "YOU WIN!" if self.win else "GAME OVER"
        text = self.font_large.render(message, True, self.COLOR_WHITE)
        text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
        self.screen.blit(text, text_rect)

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
        assert not trunc
        assert isinstance(info, dict)

        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # This block will not be executed by the grading system but is useful for testing
    # To run this, you'll need to `pip install pygame`
    # It's recommended to run this with the original os.environ setting commented out
    # os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    
    # Re-enable display for local testing
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode='rgb_array')
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption("Tetris-like Puzzle Game")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    # To solve the issue of held keys causing rapid actions, we track press events
    last_up_pressed = False
    last_space_pressed = False
    last_shift_pressed = False

    while running:
        movement = 0  # No-op
        space_action = 0
        shift_action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        
        # Handle single-press actions (rotation, hard drop)
        current_up_pressed = keys[pygame.K_UP]
        if current_up_pressed and not last_up_pressed:
            movement = 1
        last_up_pressed = current_up_pressed

        current_space_pressed = keys[pygame.K_SPACE]
        if current_space_pressed and not last_space_pressed:
            space_action = 1
        last_space_pressed = current_space_pressed
        
        current_shift_pressed = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        if current_shift_pressed and not last_shift_pressed:
            shift_action = 1
        last_shift_pressed = current_shift_pressed

        # Handle continuous-press actions (movement, soft drop)
        if movement == 0: # Only if not rotating
            if keys[pygame.K_DOWN]:
                movement = 2
            elif keys[pygame.K_LEFT]:
                movement = 3
            elif keys[pygame.K_RIGHT]:
                movement = 4

        action = [movement, space_action, shift_action]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0
            last_up_pressed = False
            last_space_pressed = False
            last_shift_pressed = False


        clock.tick(30)  # Run at 30 FPS

    env.close()