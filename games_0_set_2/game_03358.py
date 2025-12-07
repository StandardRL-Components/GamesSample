
# Generated: 2025-08-27T23:07:27.163831
# Source Brief: brief_03358.md
# Brief Index: 3358

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move, ↑↓ to rotate. Hold space for soft drop, press shift for hard drop."
    )

    game_description = (
        "Clear lines of colored blocks on a grid to reach a target score before the board fills up. A fast-paced strategic block placement puzzle game."
    )

    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 20
    BLOCK_SIZE = 18
    GRID_LINE_WIDTH = 1

    # --- Colors ---
    COLOR_BG = (20, 20, 30)
    COLOR_GRID = (40, 40, 50)
    COLOR_UI_TEXT = (220, 220, 220)
    COLOR_UI_HEADER = (180, 180, 190)
    COLOR_WHITE = (255, 255, 255)

    PIECE_COLORS = [
        (0, 240, 240),  # I (Cyan)
        (240, 240, 0),  # O (Yellow)
        (160, 0, 240),  # T (Purple)
        (0, 0, 240),    # J (Blue)
        (240, 160, 0),  # L (Orange)
        (0, 240, 0),    # S (Green)
        (240, 0, 0),    # Z (Red)
    ]

    PIECE_SHAPES = [
        [[1, 1, 1, 1]],  # I
        [[1, 1], [1, 1]],  # O
        [[0, 1, 0], [1, 1, 1]],  # T
        [[1, 0, 0], [1, 1, 1]],  # J
        [[0, 0, 1], [1, 1, 1]],  # L
        [[0, 1, 1], [1, 1, 0]],  # S
        [[1, 1, 0], [0, 1, 1]],  # Z
    ]
    
    WIN_CONDITION_LINES = 10
    MAX_STEPS = 3600 # 2 minutes at 30fps

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

        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 22)

        self.grid_render_pos = (
            (self.SCREEN_WIDTH - self.GRID_WIDTH * self.BLOCK_SIZE) // 2 - 100,
            (self.SCREEN_HEIGHT - self.GRID_HEIGHT * self.BLOCK_SIZE) // 2
        )

        self.last_action = self.action_space.sample()
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        self.piece_queue = deque()
        self._fill_piece_queue()
        self._spawn_piece()

        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.win = False

        self.fall_speed = 0.5  # blocks per second
        self.fall_counter = 0.0

        self.move_timer = 0
        self.move_cooldown = 4 # frames
        self.rotate_timer = 0
        self.rotate_cooldown = 6 # frames

        self.clear_animation_state = None # (rows, timer)
        self.last_action = np.array([0, 0, 0])

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        terminated = False
        
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if self.clear_animation_state:
            self.clear_animation_state = (self.clear_animation_state[0], self.clear_animation_state[1] - 1)
            if self.clear_animation_state[1] <= 0:
                self._finish_line_clear()
            
            return self._get_observation(), reward, terminated, False, self._get_info()

        # --- Action Handling ---
        # Handle one-shot hard drop
        if shift_held and not self.last_action[2]: # shift was just pressed
            reward += self._hard_drop()
            # Sound: Hard drop thud

        # Handle movement and rotation with cooldowns
        self.move_timer = max(0, self.move_timer - 1)
        self.rotate_timer = max(0, self.rotate_timer - 1)

        if self.move_timer == 0:
            if movement == 3: # Left
                if self._move(-1):
                    reward -= 0.01
                    self.move_timer = self.move_cooldown
            elif movement == 4: # Right
                if self._move(1):
                    reward -= 0.01
                    self.move_timer = self.move_cooldown
        
        if self.rotate_timer == 0:
            if movement == 1: # Up -> Rotate CW
                if self._rotate(1): self.rotate_timer = self.rotate_cooldown
            elif movement == 2: # Down -> Rotate CCW
                if self._rotate(-1): self.rotate_timer = self.rotate_cooldown

        self.last_action = action

        # --- Game Logic Update ---
        # Gravity
        fall_rate = self.fall_speed / 30.0  # Fall speed per frame (at 30fps)
        if space_held:
            fall_rate *= 10 # Soft drop
        
        self.fall_counter += fall_rate

        if self.fall_counter >= 1.0:
            moved_down = 0
            while self.fall_counter >= 1.0:
                if self._is_valid_position(offset_y=1):
                    self.current_piece['y'] += 1
                    moved_down += 1
                    self.fall_counter -= 1.0
                else:
                    self.fall_counter = 0.0
                    line_clear_reward, lines = self._lock_piece()
                    reward += line_clear_reward
                    if lines > 0: # Pause for animation
                        return self._get_observation(), reward, terminated, False, self._get_info()
                    break
            if moved_down > 0:
                reward += 0.1 * moved_down
        
        # --- Check Termination ---
        terminated = self.game_over or self.win or self.steps >= self.MAX_STEPS
        if terminated:
            if self.win:
                reward += 100
            elif self.game_over:
                reward += -100

        self.steps += 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines": self.lines_cleared}

    def _fill_piece_queue(self):
        bag = list(range(len(self.PIECE_SHAPES)))
        self.np_random.shuffle(bag)
        for i in bag:
            self.piece_queue.append(i)

    def _spawn_piece(self):
        if not self.piece_queue:
            self._fill_piece_queue()
        
        piece_index = self.piece_queue.popleft()
        shape = self.PIECE_SHAPES[piece_index]
        
        self.current_piece = {
            'shape': shape,
            'color_index': piece_index + 1,
            'x': self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'rotation': 0
        }
        
        if not self._is_valid_position():
            self.game_over = True
            # Sound: Game over
    
    def _is_valid_position(self, piece=None, offset_x=0, offset_y=0):
        if piece is None:
            piece = self.current_piece
        
        shape = piece['shape']
        px, py = piece['x'] + offset_x, piece['y'] + offset_y
        
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = px + x, py + y
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT):
                        return False
                    if self.grid[grid_y, grid_x] != 0:
                        return False
        return True

    def _move(self, dx):
        if self._is_valid_position(offset_x=dx):
            self.current_piece['x'] += dx
            # Sound: Piece move
            return True
        return False

    def _rotate(self, direction):
        original_shape = self.current_piece['shape']
        
        if direction == 1: # Clockwise
            new_shape = [list(row) for row in zip(*original_shape[::-1])]
        else: # Counter-clockwise
            new_shape = [list(row) for row in zip(*original_shape)][::-1]

        test_piece = self.current_piece.copy()
        test_piece['shape'] = new_shape

        # Wall kick logic
        for kick_x in [0, -1, 1, -2, 2]:
            if self._is_valid_position(test_piece, offset_x=kick_x):
                self.current_piece['shape'] = new_shape
                self.current_piece['x'] += kick_x
                # Sound: Piece rotate
                return True
        return False

    def _hard_drop(self):
        dy = 0
        while self._is_valid_position(offset_y=dy + 1):
            dy += 1
        self.current_piece['y'] += dy
        reward, _ = self._lock_piece()
        return reward + (0.1 * dy)

    def _lock_piece(self):
        shape = self.current_piece['shape']
        px, py = self.current_piece['x'], self.current_piece['y']
        
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[py + y, px + x] = self.current_piece['color_index']
        
        # Sound: Piece lock
        reward, lines_cleared = self._check_lines()
        
        if not self.game_over and self.clear_animation_state is None:
            self._spawn_piece()
            
        return reward, lines_cleared

    def _check_lines(self):
        full_rows = [r for r in range(self.GRID_HEIGHT) if np.all(self.grid[r, :] != 0)]
        
        if not full_rows:
            return 0, 0
        
        # Start animation
        self.clear_animation_state = (full_rows, 6) # 6 frames animation
        # Sound: Line clear
        
        num_cleared = len(full_rows)
        self.lines_cleared += num_cleared
        
        # Score for lines
        self.score += [0, 100, 300, 500, 800][num_cleared]
        
        # Update fall speed
        self.fall_speed = 0.5 + (self.lines_cleared // 5) * 0.1
        
        if self.lines_cleared >= self.WIN_CONDITION_LINES:
            self.win = True
            # Sound: Win jingle

        # Reward for lines
        reward = {1: 1, 2: 2, 3: 4, 4: 8}.get(num_cleared, 0)
        return reward, num_cleared

    def _finish_line_clear(self):
        full_rows = self.clear_animation_state[0]
        
        for r in sorted(full_rows, reverse=True):
            self.grid[1:r+1, :] = self.grid[0:r, :]
            self.grid[0, :] = 0
        
        self.clear_animation_state = None
        self._spawn_piece()

    def _draw_block(self, surface, color_index, x, y, size, alpha=255, outline_width=2):
        if color_index == 0: return
        color = self.PIECE_COLORS[color_index - 1]
        
        rect = pygame.Rect(x, y, size, size)
        
        # Main color
        shape_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        shape_surf.fill((*color, alpha))
        
        # Lighter inner part
        inner_color = tuple(min(255, c + 50) for c in color)
        pygame.draw.rect(shape_surf, (*inner_color, alpha), (outline_width, outline_width, size - 2*outline_width, size - 2*outline_width))
        
        # Darker outline
        outline_color = tuple(max(0, c - 50) for c in color)
        pygame.draw.rect(shape_surf, (*outline_color, alpha), (0, 0, size, size), outline_width)
        
        surface.blit(shape_surf, rect)

    def _render_game(self):
        gx, gy = self.grid_render_pos
        
        # Draw grid background and lines
        grid_rect = pygame.Rect(gx, gy, self.GRID_WIDTH * self.BLOCK_SIZE, self.GRID_HEIGHT * self.BLOCK_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_GRID, grid_rect)
        for i in range(self.GRID_WIDTH + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (gx + i * self.BLOCK_SIZE, gy), (gx + i * self.BLOCK_SIZE, gy + self.GRID_HEIGHT * self.BLOCK_SIZE), self.GRID_LINE_WIDTH)
        for i in range(self.GRID_HEIGHT + 1):
            pygame.draw.line(self.screen, self.COLOR_BG, (gx, gy + i * self.BLOCK_SIZE), (gx + self.GRID_WIDTH * self.BLOCK_SIZE, gy + i * self.BLOCK_SIZE), self.GRID_LINE_WIDTH)

        # Draw locked blocks
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.grid[r, c] != 0:
                    self._draw_block(self.screen, self.grid[r, c], gx + c * self.BLOCK_SIZE, gy + r * self.BLOCK_SIZE, self.BLOCK_SIZE)

        # Draw line clear animation
        if self.clear_animation_state:
            rows, timer = self.clear_animation_state
            # Flash white
            if (timer // 2) % 2 == 1:
                for r in rows:
                    flash_rect = pygame.Rect(gx, gy + r * self.BLOCK_SIZE, self.GRID_WIDTH * self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(self.screen, self.COLOR_WHITE, flash_rect)
            return # Pause rendering of falling piece during animation

        if self.game_over: return

        # Draw ghost piece
        ghost_y = self.current_piece['y']
        while self._is_valid_position(offset_y=ghost_y - self.current_piece['y'] + 1):
            ghost_y += 1
        
        shape = self.current_piece['shape']
        px = self.current_piece['x']
        color = self.PIECE_COLORS[self.current_piece['color_index'] - 1]
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(gx + (px + x) * self.BLOCK_SIZE, gy + (ghost_y + y) * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    pygame.draw.rect(self.screen, color, rect, 2, border_radius=2)

        # Draw current piece
        py = self.current_piece['y']
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self._draw_block(self.screen, self.current_piece['color_index'], gx + (px + x) * self.BLOCK_SIZE, gy + (py + y) * self.BLOCK_SIZE, self.BLOCK_SIZE)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"{self.score}", True, self.COLOR_UI_TEXT)
        score_header = self.font_small.render("SCORE", True, self.COLOR_UI_HEADER)
        self.screen.blit(score_header, (30, 30))
        self.screen.blit(score_text, (30, 55))

        # Lines
        lines_text = self.font_large.render(f"{self.lines_cleared} / {self.WIN_CONDITION_LINES}", True, self.COLOR_UI_TEXT)
        lines_header = self.font_small.render("LINES", True, self.COLOR_UI_HEADER)
        self.screen.blit(lines_header, (self.SCREEN_WIDTH - 150, 30))
        self.screen.blit(lines_text, (self.SCREEN_WIDTH - 150, 55))

        # Next Piece
        next_header = self.font_small.render("NEXT", True, self.COLOR_UI_HEADER)
        self.screen.blit(next_header, (self.SCREEN_WIDTH - 150, 120))
        
        if self.piece_queue:
            next_piece_index = self.piece_queue[0]
            shape = self.PIECE_SHAPES[next_piece_index]
            w, h = len(shape[0]), len(shape)
            start_x = self.SCREEN_WIDTH - 150 + (100 - w * self.BLOCK_SIZE) / 2
            start_y = 150 + (80 - h * self.BLOCK_SIZE) / 2
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.screen, next_piece_index + 1, start_x + x * self.BLOCK_SIZE, start_y + y * self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        # Game Over / Win Text
        if self.game_over:
            text = self.font_large.render("GAME OVER", True, (255, 50, 50))
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)
        elif self.win:
            text = self.font_large.render("YOU WIN!", True, (50, 255, 50))
            text_rect = text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(text, text_rect)

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

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Gymnasium Tetris")
    clock = pygame.time.Clock()
    
    action = np.array([0, 0, 0]) # [movement, space, shift]

    while not terminated:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        
        # Reset actions
        action[0] = 0 # No movement
        action[1] = 0 # Space released
        action[2] = 0 # Shift released
        
        # Map keys to actions
        if keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        elif keys[pygame.K_UP]:
            action[0] = 1 # Rotate CW
        elif keys[pygame.K_DOWN]:
            action[0] = 2 # Rotate CCW

        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)

        # Render the observation from the environment to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Run at 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Lines: {info['lines']}")
    env.close()
    pygame.quit()