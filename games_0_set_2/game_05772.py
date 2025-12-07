
# Generated: 2025-08-28T06:04:06.989522
# Source Brief: brief_05772.md
# Brief Index: 5772

        
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
        "Controls: ←→ to move, ↓ for soft drop. ↑ or Space to rotate. Shift for hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A fast-paced falling block puzzle game. Rotate and place tetrominoes to clear lines, "
        "score points, and prevent the stack from reaching the top. The game gets faster as you clear more lines!"
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.CELL_SIZE = 18
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH * self.CELL_SIZE) // 2
        self.GRID_Y = (self.HEIGHT - self.GRID_HEIGHT * self.CELL_SIZE) // 2
        self.MAX_STEPS = 1000
        self.WIN_CONDITION_LINES = 15

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_GRID = (40, 40, 50)
        self.COLOR_GRID_LINES = (60, 60, 70)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_FLASH = (255, 255, 255)
        
        self.TETROMINO_COLORS = [
            (0, 0, 0),       # 0: Empty
            (255, 50, 50),   # 1: Z (Red)
            (50, 255, 50),   # 2: S (Green)
            (50, 50, 255),   # 3: J (Blue)
            (255, 150, 50),  # 4: L (Orange)
            (255, 255, 50),  # 5: O (Yellow)
            (150, 50, 255),  # 6: T (Purple)
            (50, 255, 255),  # 7: I (Cyan)
        ]

        # Tetromino shapes (indices match colors)
        self.TETROMINOES = [
            [], # 0: Empty
            [[[1, 1, 0], [0, 1, 1], [0, 0, 0]]], # 1: Z
            [[[0, 1, 1], [1, 1, 0], [0, 0, 0]]], # 2: S
            [[[1, 0, 0], [1, 1, 1], [0, 0, 0]]], # 3: J
            [[[0, 0, 1], [1, 1, 1], [0, 0, 0]]], # 4: L
            [[[1, 1], [1, 1]]],                 # 5: O
            [[[0, 1, 0], [1, 1, 1], [0, 0, 0]]], # 6: T
            [[[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]]], # 7: I
        ]
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_ui = pygame.font.SysFont("Consolas", 18)
        
        # Game state variables are initialized in reset()
        self.grid = None
        self.current_piece = None
        self.next_piece = None
        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_time = 0
        self.base_fall_speed = 0.5
        self.fall_speed = 0.5
        self.last_action_states = {'space': 0, 'shift': 0, 'up': 0}
        self.line_clear_anim_timer = 0
        self.lines_being_cleared = []
        
        # Initialize state variables
        self.reset()
        self.validate_implementation()
    
    def _get_tetromino_shape(self, type_idx, rotation):
        base_shape = self.TETROMINOES[type_idx][0]
        if type_idx == 5: # O-shape doesn't rotate
            return base_shape
        
        rotated_shape = np.array(base_shape)
        # Use np.rot90 for efficient rotation
        rotated_shape = np.rot90(rotated_shape, k=-rotation) # k is number of 90 deg rotations
        
        return rotated_shape.tolist()

    def _new_piece(self):
        self.current_piece = self.next_piece
        
        next_type = self.np_random.integers(1, len(self.TETROMINOES))
        self.next_piece = {
            'type': next_type,
            'rotation': 0,
            'x': self.GRID_WIDTH // 2 - 1,
            'y': 0,
            'color_idx': next_type
        }
        
        if self.current_piece and not self._is_valid_position(self.current_piece):
            self.game_over = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=int)
        
        next_type = self.np_random.integers(1, len(self.TETROMINOES))
        self.next_piece = {
            'type': next_type, 'rotation': 0, 'x': self.GRID_WIDTH // 2 - 1, 'y': 0, 'color_idx': next_type
        }
        self._new_piece()

        self.score = 0
        self.lines_cleared = 0
        self.steps = 0
        self.game_over = False
        self.fall_time = 0
        self.base_fall_speed = 0.5
        self.fall_speed = self.base_fall_speed
        self.last_action_states = {'space': 0, 'shift': 0, 'up': 0}
        self.line_clear_anim_timer = 0
        self.lines_being_cleared = []
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        # Update game logic
        self.steps += 1
        reward = -0.01  # Small penalty for each step to encourage speed
        terminated = False

        if self.game_over:
            reward -= 100
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Pause game logic during line clear animation for visual effect
        if self.line_clear_anim_timer > 0:
            self.line_clear_anim_timer -= 1
            if self.line_clear_anim_timer == 0:
                self._remove_cleared_lines()
            return self._get_observation(), reward, terminated, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean
        
        # Handle discrete actions (on rising edge to prevent rapid fire)
        up_pressed = (movement == 1) and (self.last_action_states['up'] == 0)
        space_pressed = space_held and (self.last_action_states['space'] == 0)
        shift_pressed = shift_held and (self.last_action_states['shift'] == 0)

        self.last_action_states = {'up': movement == 1, 'space': space_held, 'shift': shift_held}
        
        # --- Action Handling ---
        if up_pressed: self._rotate_piece(1) # Rotate clockwise
        if space_pressed: self._rotate_piece(-1) # Rotate counter-clockwise
        
        if movement == 3: self._move(-1, 0) # Move left
        elif movement == 4: self._move(1, 0) # Move right

        if shift_pressed:
            self._hard_drop()
            # Sound placeholder: pygame.mixer.Sound("hard_drop.wav").play()
            reward += self._lock_piece()
        else:
            # --- Game Tick / Fall Logic ---
            time_delta = 1.0 / 30.0 # Assuming 30 FPS for auto_advance
            self.fall_time += time_delta
            
            soft_drop = (movement == 2)
            current_fall_speed = self.fall_speed / 10.0 if soft_drop else self.fall_speed

            if self.fall_time >= current_fall_speed:
                self.fall_time = 0
                if not self._move(0, 1): # If move down fails, lock piece
                    reward += self._lock_piece()

        # --- Check Termination Conditions ---
        terminated = self._check_termination()
        if terminated and self.game_over:
            reward -= 100
        elif terminated and self.lines_cleared >= self.WIN_CONDITION_LINES:
            reward += 100
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _check_termination(self):
        return self.game_over or self.lines_cleared >= self.WIN_CONDITION_LINES or self.steps >= self.MAX_STEPS

    def _move(self, dx, dy):
        if self.current_piece and self._is_valid_position(self.current_piece, dx, dy):
            self.current_piece['x'] += dx
            self.current_piece['y'] += dy
            return True
        return False

    def _rotate_piece(self, direction):
        if not self.current_piece: return
        
        original_rotation = self.current_piece['rotation']
        self.current_piece['rotation'] = (self.current_piece['rotation'] + direction) % 4
        
        # Wall kick logic
        if not self._is_valid_position(self.current_piece):
            for kick_x, kick_y in [(1, 0), (-1, 0), (2, 0), (-2, 0), (0, -1)]: # Basic wall kicks
                if self._is_valid_position(self.current_piece, kick_x, kick_y):
                    self.current_piece['x'] += kick_x
                    self.current_piece['y'] += kick_y
                    return
            self.current_piece['rotation'] = original_rotation # Revert if all kicks fail

    def _is_valid_position(self, piece, dx=0, dy=0):
        shape = self._get_tetromino_shape(piece['type'], piece['rotation'])
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x, new_y = piece['x'] + x + dx, piece['y'] + y + dy
                    if not (0 <= new_x < self.GRID_WIDTH and 0 <= new_y < self.GRID_HEIGHT): return False
                    if self.grid[int(new_y), int(new_x)] != 0: return False
        return True

    def _hard_drop(self):
        if not self.current_piece: return
        while self._is_valid_position(self.current_piece, 0, 1):
            self.current_piece['y'] += 1

    def _lock_piece(self):
        reward = 0
        shape = self._get_tetromino_shape(self.current_piece['type'], self.current_piece['rotation'])
        holes_before = self._calculate_holes()

        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = self.current_piece['x'] + x, self.current_piece['y'] + y
                    if 0 <= grid_y < self.GRID_HEIGHT and 0 <= grid_x < self.GRID_WIDTH:
                        self.grid[int(grid_y), int(grid_x)] = self.current_piece['color_idx']
        
        holes_after = self._calculate_holes()
        reward -= max(0, holes_after - holes_before) * 0.2

        cleared_count = self._check_for_line_clears()
        if cleared_count > 0:
            reward += {1: 1, 2: 3, 3: 7, 4: 15}.get(cleared_count, 15)
            self.score += [0, 100, 300, 500, 800][cleared_count]
            self.lines_cleared += cleared_count
            self.fall_speed = max(0.05, self.base_fall_speed * (0.9 ** (self.lines_cleared // 5)))

        self._new_piece()
        # Sound placeholder: pygame.mixer.Sound("lock.wav").play()
        return reward

    def _check_for_line_clears(self):
        self.lines_being_cleared = [y for y in range(self.GRID_HEIGHT) if np.all(self.grid[y] != 0)]
        if self.lines_being_cleared:
            self.line_clear_anim_timer = 5 # Animate for 5 frames
            # Sound placeholder: pygame.mixer.Sound("clear.wav").play()
        return len(self.lines_being_cleared)

    def _remove_cleared_lines(self):
        self.grid = np.delete(self.grid, self.lines_being_cleared, axis=0)
        new_rows = np.zeros((len(self.lines_being_cleared), self.GRID_WIDTH), dtype=int)
        self.grid = np.vstack([new_rows, self.grid])
        self.lines_being_cleared = []

    def _calculate_holes(self):
        return np.sum((np.cumsum(self.grid, axis=0) > 0) & (self.grid == 0))

    def _calculate_reward(self): return 0 # Handled in step
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        pygame.draw.rect(self.screen, self.COLOR_GRID, (self.GRID_X, self.GRID_Y, self.GRID_WIDTH * self.CELL_SIZE, self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                if self.grid[y, x] != 0: self._draw_cell(x, y, self.grid[y, x])
        
        if self.current_piece and not self.game_over:
            ghost_piece = self.current_piece.copy()
            while self._is_valid_position(ghost_piece, 0, 1): ghost_piece['y'] += 1
            self._draw_piece(ghost_piece, is_ghost=True)
            self._draw_piece(self.current_piece)
            
        if self.line_clear_anim_timer > 0:
            for y in self.lines_being_cleared:
                pygame.draw.rect(self.screen, self.COLOR_FLASH, (self.GRID_X, self.GRID_Y + y * self.CELL_SIZE, self.GRID_WIDTH * self.CELL_SIZE, self.CELL_SIZE))
        
        for x in range(self.GRID_WIDTH + 1):
            px = self.GRID_X + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (px, self.GRID_Y), (px, self.GRID_Y + self.GRID_HEIGHT * self.CELL_SIZE))
        for y in range(self.GRID_HEIGHT + 1):
            py = self.GRID_Y + y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID_LINES, (self.GRID_X, py), (self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE, py))

    def _draw_piece(self, piece, is_ghost=False):
        shape = self._get_tetromino_shape(piece['type'], piece['rotation'])
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell: self._draw_cell(piece['x'] + x, piece['y'] + y, piece['color_idx'], is_ghost)
    
    def _draw_cell(self, grid_x, grid_y, color_idx, is_ghost=False):
        px, py = int(self.GRID_X + grid_x * self.CELL_SIZE), int(self.GRID_Y + grid_y * self.CELL_SIZE)
        color = self.TETROMINO_COLORS[color_idx]
        
        if is_ghost:
            pygame.draw.rect(self.screen, color, (px, py, self.CELL_SIZE, self.CELL_SIZE), 1)
        else:
            main_rect = pygame.Rect(px, py, self.CELL_SIZE, self.CELL_SIZE)
            highlight = tuple(min(255, c + 40) for c in color)
            shadow = tuple(max(0, c - 40) for c in color)
            pygame.draw.rect(self.screen, shadow, main_rect)
            pygame.draw.rect(self.screen, color, main_rect.inflate(-2, -2))
            pygame.gfxdraw.aacircle(self.screen, px + self.CELL_SIZE//2, py + self.CELL_SIZE//2, self.CELL_SIZE//4, highlight)

    def _render_ui(self):
        ui_x = self.GRID_X + self.GRID_WIDTH * self.CELL_SIZE + 20
        self.screen.blit(self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT), (ui_x, self.GRID_Y))
        self.screen.blit(self.font_ui.render(f"LINES: {self.lines_cleared}", True, self.COLOR_TEXT), (ui_x, self.GRID_Y + 30))
        self.screen.blit(self.font_ui.render("NEXT:", True, self.COLOR_TEXT), (ui_x, self.GRID_Y + 80))
        
        if self.next_piece:
            shape = self._get_tetromino_shape(self.next_piece['type'], 0)
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell: self._draw_cell(
                        (ui_x - self.GRID_X) / self.CELL_SIZE + 2 + x, 
                        (self.GRID_Y + 110 - self.GRID_Y) / self.CELL_SIZE + y, 
                        self.next_piece['color_idx'])

        msg = "GAME OVER" if self.game_over else "YOU WIN!" if self._check_termination() and not self.game_over else None
        if msg:
            text_surf = self.font_main.render(msg, True, self.COLOR_FLASH)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            s = pygame.Surface(text_rect.inflate(20,20).size, pygame.SRCALPHA); s.fill((0,0,0,180))
            self.screen.blit(s, text_rect.inflate(20,20))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "lines_cleared": self.lines_cleared}

    def validate_implementation(self):
        assert self.action_space.shape == (3,) and self.action_space.nvec.tolist() == [5, 2, 2]
        obs = self._get_observation()
        assert obs.shape == (400, 640, 3) and obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3) and isinstance(info, dict)
        obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert obs.shape == (400, 640, 3) and isinstance(reward, (int, float))
        assert isinstance(term, bool) and not trunc and isinstance(info, dict)
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Gymnasium Block Puzzle")
    clock = pygame.time.Clock()
    running, total_reward = True, 0

    while running:
        movement, space, shift = 0, 0, 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset(); total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        obs, reward, terminated, truncated, info = env.step([movement, space, shift])
        total_reward += reward

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Score: {info['score']}, Reward: {total_reward:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000)
            obs, info = env.reset(); total_reward = 0

        clock.tick(30)
    pygame.quit()