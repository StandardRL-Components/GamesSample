
# Generated: 2025-08-28T02:44:08.999441
# Source Brief: brief_01795.md
# Brief Index: 1795

        
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
        "Controls: ↑ to rotate, ←→ to move, ↓ to soft drop. Hold shift to hold piece and press space to hard drop."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fast-paced arcade puzzle game. Rotate and drop falling blocks to clear lines and score points before the stack reaches the top."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # Game constants
        self.GRID_WIDTH, self.GRID_HEIGHT = 10, 20
        self.BLOCK_SIZE = 18
        self.PLAYFIELD_WIDTH = self.GRID_WIDTH * self.BLOCK_SIZE
        self.PLAYFIELD_HEIGHT = self.GRID_HEIGHT * self.BLOCK_SIZE
        self.PLAYFIELD_X = (self.WIDTH - self.PLAYFIELD_WIDTH) // 2
        self.PLAYFIELD_Y = (self.HEIGHT - self.PLAYFIELD_HEIGHT) // 2
        self.WIN_CONDITION = 20
        self.MAX_STEPS = 10000
        self.SOFT_DROP_SPEEDUP = 5

        # Colors
        self.COLOR_BG = (22, 22, 34)
        self.COLOR_GRID = (40, 40, 60)
        self.COLOR_UI_TEXT = (230, 230, 255)
        self.COLOR_WHITE = (255, 255, 255)
        self.TETROMINO_COLORS = [
            (0, 240, 240),  # I (Cyan)
            (240, 240, 0),  # O (Yellow)
            (160, 0, 240),  # T (Purple)
            (0, 240, 0),    # S (Green)
            (240, 0, 0),    # Z (Red)
            (0, 0, 240),    # J (Blue)
            (240, 160, 0),  # L (Orange)
        ]

        # Tetromino shapes
        self.TETROMINOES = [
            [[1, 1, 1, 1]],  # I
            [[1, 1], [1, 1]],  # O
            [[0, 1, 0], [1, 1, 1]],  # T
            [[0, 1, 1], [1, 1, 0]],  # S
            [[1, 1, 0], [0, 1, 1]],  # Z
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 0, 1], [1, 1, 1]],  # L
        ]
        
        # Fonts
        self.font_main = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)

        # Game state variables (initialized in reset)
        self.grid = None
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.piece_bag = []
        self.current_piece = None
        self.next_piece = None
        self.held_piece = None
        self.can_hold = True
        self.fall_rate = 30  # Frames per grid cell drop
        self.fall_counter = 0
        self.line_clear_animation = []
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT + 4), dtype=int)
        self.steps = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        self.piece_bag = list(range(len(self.TETROMINOES)))
        self.np_random.shuffle(self.piece_bag)
        
        self.held_piece = None
        self.can_hold = True
        
        self.fall_rate = 30
        self.fall_counter = 0
        self.line_clear_animation = []

        self._new_piece()
        self._new_piece() # Once for current, once for next

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty per step to encourage speed

        if self.game_over:
            return self._get_observation(), reward, True, False, self._get_info()
        
        self.steps += 1
        
        # Handle line clear animation delay
        if self.line_clear_animation:
            self.line_clear_animation[0] -= 1
            if self.line_clear_animation[0] <= 0:
                self._finish_line_clear()
        else:
            # Process input and game logic only if not in an animation
            self._handle_input(action)
            reward += self._update_fall_logic()

        terminated = self.game_over or self.lines_cleared >= self.WIN_CONDITION or self.steps >= self.MAX_STEPS
        
        if terminated:
            if self.lines_cleared >= self.WIN_CONDITION:
                reward += 100  # Win bonus
            elif self.game_over:
                reward += -50 # Lose penalty

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self._rotate_piece() # Up: Rotate
        if movement == 3: self._move_piece(-1) # Left
        if movement == 4: self._move_piece(1)  # Right
        if movement == 2: self.fall_counter += self.SOFT_DROP_SPEEDUP # Down: Soft drop

        if space_held:
            reward = self._hard_drop()
            self.fall_counter = self.fall_rate # Force lock on next frame
            return reward

        if shift_held and self.can_hold:
            self._hold_piece()
        
        return 0

    def _update_fall_logic(self):
        self.fall_counter += 1
        if self.fall_counter >= self.fall_rate:
            self.fall_counter = 0
            self.current_piece['y'] += 1
            if self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
                self.current_piece['y'] -= 1
                # # Sound effect placeholder:
                # pygame.mixer.Sound("sounds/lock.wav").play()
                return self._lock_piece()
        return 0

    def _new_piece(self):
        if not self.piece_bag:
            self.piece_bag = list(range(len(self.TETROMINOES)))
            self.np_random.shuffle(self.piece_bag)

        piece_index = self.piece_bag.pop(0)
        shape = self.TETROMINOES[piece_index]
        
        if self.current_piece is None:
            self.current_piece = self._create_piece_data(piece_index, shape)
        else:
            self.next_piece = self._create_piece_data(piece_index, shape)

        if self._check_collision(self.current_piece['shape'], (self.current_piece['x'], self.current_piece['y'])):
            self.game_over = True

    def _create_piece_data(self, index, shape):
        return {
            'index': index,
            'shape': shape,
            'x': self.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0,
            'color': self.TETROMINO_COLORS[index]
        }

    def _lock_piece(self):
        for r, row in enumerate(self.current_piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    grid_x = self.current_piece['x'] + c
                    grid_y = self.current_piece['y'] + r
                    if 0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT + 4:
                        self.grid[grid_x, grid_y] = self.current_piece['index'] + 1
        
        reward = self._clear_lines()

        self.current_piece = self.next_piece
        self._new_piece()

        self.can_hold = True
        return reward

    def _clear_lines(self):
        lines_to_clear = []
        for r in range(self.GRID_HEIGHT + 4):
            if np.all(self.grid[:, r]):
                lines_to_clear.append(r)
        
        if lines_to_clear:
            # # Sound effect placeholder:
            # pygame.mixer.Sound("sounds/clear.wav").play()
            self.line_clear_animation = [10, lines_to_clear] # 10 frames duration
            for r in lines_to_clear:
                self.grid[:, r] = -1 # Mark for animation

            num_cleared = len(lines_to_clear)
            self.score += [0, 100, 300, 600, 1000][num_cleared]
            reward_map = {1: 1, 2: 3, 3: 6, 4: 10}
            return reward_map.get(num_cleared, 0)
        return 0

    def _finish_line_clear(self):
        lines_cleared_indices = self.line_clear_animation[1]
        num_cleared = len(lines_cleared_indices)
        
        # Shift grid down
        for r in sorted(lines_cleared_indices, reverse=True):
            self.grid[:, 1:r+1] = self.grid[:, 0:r]
            self.grid[:, 0] = 0

        self.lines_cleared += num_cleared
        self.score += 50 * num_cleared # Bonus score
        
        # Update fall speed every 5 lines
        speed_increase_tiers = self.lines_cleared // 5
        self.fall_rate = max(5, 30 - speed_increase_tiers)

        self.line_clear_animation = []

    def _move_piece(self, dx):
        new_x = self.current_piece['x'] + dx
        if not self._check_collision(self.current_piece['shape'], (new_x, self.current_piece['y'])):
            self.current_piece['x'] = new_x
            # # Sound effect placeholder:
            # pygame.mixer.Sound("sounds/move.wav").play()

    def _rotate_piece(self):
        original_shape = self.current_piece['shape']
        rotated_shape = list(zip(*original_shape[::-1]))
        
        # Basic wall kick tests
        offsets = [0, 1, -1, 2, -2]
        for offset in offsets:
            new_x = self.current_piece['x'] + offset
            if not self._check_collision(rotated_shape, (new_x, self.current_piece['y'])):
                self.current_piece['shape'] = rotated_shape
                self.current_piece['x'] = new_x
                # # Sound effect placeholder:
                # pygame.mixer.Sound("sounds/rotate.wav").play()
                return

    def _hard_drop(self):
        ghost_y = self._get_ghost_y()
        self.current_piece['y'] = ghost_y
        return self._lock_piece()

    def _hold_piece(self):
        self.can_hold = False
        if self.held_piece is None:
            self.held_piece = self._create_piece_data(self.current_piece['index'], self.TETROMINOES[self.current_piece['index']])
            self.current_piece = self.next_piece
            self._new_piece()
        else:
            self.current_piece, self.held_piece = self.held_piece, self._create_piece_data(self.current_piece['index'], self.TETROMINOES[self.current_piece['index']])
            self.current_piece['x'] = self.GRID_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
            self.current_piece['y'] = 0

    def _check_collision(self, shape, pos):
        px, py = pos
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    grid_x, grid_y = px + c, py + r
                    if not (0 <= grid_x < self.GRID_WIDTH and 0 <= grid_y < self.GRID_HEIGHT + 4):
                        return True
                    if self.grid[grid_x, grid_y] > 0:
                        return True
        return False

    def _get_ghost_y(self):
        y = self.current_piece['y']
        while not self._check_collision(self.current_piece['shape'], (self.current_piece['x'], y + 1)):
            y += 1
        return y

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_playfield()
        self._render_pieces()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_playfield(self):
        # Draw border
        border_rect = pygame.Rect(self.PLAYFIELD_X - 2, self.PLAYFIELD_Y - 2, self.PLAYFIELD_WIDTH + 4, self.PLAYFIELD_HEIGHT + 4)
        pygame.draw.rect(self.screen, self.COLOR_GRID, border_rect, 2, 5)

        # Draw grid lines
        for x in range(1, self.GRID_WIDTH):
            start = (self.PLAYFIELD_X + x * self.BLOCK_SIZE, self.PLAYFIELD_Y)
            end = (self.PLAYFIELD_X + x * self.BLOCK_SIZE, self.PLAYFIELD_Y + self.PLAYFIELD_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)
        for y in range(1, self.GRID_HEIGHT):
            start = (self.PLAYFIELD_X, self.PLAYFIELD_Y + y * self.BLOCK_SIZE)
            end = (self.PLAYFIELD_X + self.PLAYFIELD_WIDTH, self.PLAYFIELD_Y + y * self.BLOCK_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end)

    def _render_pieces(self):
        # Draw settled blocks
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                cell_val = self.grid[x, y + 4]
                if cell_val > 0:
                    color = self.TETROMINO_COLORS[int(cell_val) - 1]
                    self._draw_block(x, y, color)
                elif cell_val == -1: # Line clear animation
                    self._draw_block(x, y, self.COLOR_WHITE, flash=True)

        if self.game_over: return

        # Draw ghost piece
        ghost_y = self._get_ghost_y()
        if ghost_y > self.current_piece['y']:
            for r, row in enumerate(self.current_piece['shape']):
                for c, cell in enumerate(row):
                    if cell:
                        self._draw_block(self.current_piece['x'] + c, ghost_y + r - 4, self.current_piece['color'], is_ghost=True)

        # Draw current piece
        for r, row in enumerate(self.current_piece['shape']):
            for c, cell in enumerate(row):
                if cell:
                    self._draw_block(self.current_piece['x'] + c, self.current_piece['y'] + r - 4, self.current_piece['color'])

    def _draw_block(self, grid_x, grid_y, color, is_ghost=False, flash=False):
        if grid_y < 0: return # Don't draw above playfield
        
        px = self.PLAYFIELD_X + grid_x * self.BLOCK_SIZE
        py = self.PLAYFIELD_Y + grid_y * self.BLOCK_SIZE
        rect = pygame.Rect(px, py, self.BLOCK_SIZE, self.BLOCK_SIZE)

        if is_ghost:
            pygame.draw.rect(self.screen, color, rect, 2, 3)
        else:
            if flash:
                # Pulsing effect for line clear
                pulse = abs(math.sin(self.steps * 0.5))
                color = tuple(min(255, int(c + (255 - c) * pulse)) for c in self.COLOR_GRID)

            # Main block
            pygame.gfxdraw.box(self.screen, rect, (*color, 200 if self.game_over else 255))
            # Border
            pygame.draw.rect(self.screen, tuple(c*0.5 for c in color), rect, 1)
            # Highlight
            pygame.draw.line(self.screen, self.COLOR_WHITE, (px+2, py+2), (px + self.BLOCK_SIZE - 3, py+2), 1)
            pygame.draw.line(self.screen, self.COLOR_WHITE, (px+2, py+2), (px+2, py + self.BLOCK_SIZE - 3), 1)

    def _render_ui(self):
        # Score
        self._render_text("SCORE", (40, 40), self.font_small, self.COLOR_UI_TEXT)
        self._render_text(f"{self.score:06d}", (40, 65), self.font_main, self.COLOR_WHITE)

        # Lines
        self._render_text("LINES", (40, 115), self.font_small, self.COLOR_UI_TEXT)
        self._render_text(f"{self.lines_cleared}/{self.WIN_CONDITION}", (40, 140), self.font_main, self.COLOR_WHITE)

        # Next Piece
        self._render_text("NEXT", (self.WIDTH - 100, 40), self.font_small, self.COLOR_UI_TEXT)
        if self.next_piece:
            self._render_preview_piece(self.next_piece, (self.WIDTH - 100, 85))

        # Held Piece
        self._render_text("HOLD", (self.WIDTH - 100, 180), self.font_small, self.COLOR_UI_TEXT)
        if self.held_piece:
            self._render_preview_piece(self.held_piece, (self.WIDTH - 100, 225), not self.can_hold)
            
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            msg = "GAME OVER"
            if self.lines_cleared >= self.WIN_CONDITION:
                msg = "YOU WIN!"
            self._render_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_main, self.COLOR_WHITE, center=True)

    def _render_text(self, text, pos, font, color, center=False):
        text_surf = font.render(text, True, color)
        text_rect = text_surf.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos
        self.screen.blit(text_surf, text_rect)

    def _render_preview_piece(self, piece, pos, is_dim=False):
        shape = piece['shape']
        color = piece['color']
        off_x = pos[0]
        off_y = pos[1]
        
        for r, row in enumerate(shape):
            for c, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(off_x + c * self.BLOCK_SIZE, off_y + r * self.BLOCK_SIZE, self.BLOCK_SIZE, self.BLOCK_SIZE)
                    final_color = tuple(c * 0.5 for c in color) if is_dim else color
                    pygame.gfxdraw.box(self.screen, rect, (*final_color, 255))
                    pygame.draw.rect(self.screen, tuple(c*0.5 for c in final_color), rect, 1)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lines_cleared": self.lines_cleared
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Puzzle Drop")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
            
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}, Steps: {info['steps']}")
            # Wait for a moment before auto-resetting or quitting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS
        
    env.close()